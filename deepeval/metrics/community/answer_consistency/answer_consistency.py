import asyncio
from typing import Dict, List, Optional, Tuple, Union

from deepeval.metrics import BaseMetric
from deepeval.metrics.community._consistency_utils import (
    describe_pair,
    get_runs,
    require_repeatability_evals,
    sample_pairs,
)
from deepeval.metrics.community.answer_consistency.schema import (
    AgreementScores,
    AgreementVerdict,
)
from deepeval.metrics.community.answer_consistency.template import (
    AnswerConsistencyTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    construct_verbose_logs,
    generate_with_schema_and_extract,
    initialize_model,
    prepare_measure,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.utils import get_or_create_event_loop

# (index of run a, index of run b, verdict)
_Judged = Tuple[int, int, AgreementVerdict]

_IDENTICAL = AgreementVerdict(
    reason="The runs are identical.",
    scores=AgreementScores(
        core_facts=5, specifics=5, supporting_detail=5, stance=5, format=5
    ),
)


def _same(a: str, b: str) -> bool:
    return " ".join(str(a).casefold().split()) == " ".join(
        str(b).casefold().split()
    )


class AnswerConsistencyMetric(BaseMetric):
    """How consistently a system answers the same input across repeated runs.

    Needs no expected output. Provide N >= 2 runs of the same ``input`` in
    ``metadata["runs"]`` (``actual_output`` should be one of them). A judge compares pairs of
    runs on five dimensions (core facts, specifics, supporting detail, stance, format; each 1-5)
    and the score is the weighted mean agreement, normalised to ``[0, 1]``. It measures
    agreement, not correctness: a consistently wrong system scores 1.0.

    Comparing all N(N-1)/2 pairs gets expensive, so by default a random cycle of N pairs is
    judged, which still compares every run at least twice. Set ``max_comparisons`` to change it.
    Identical runs (ignoring case and whitespace) are scored 5 on every dimension without a
    judge call.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.8,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        max_comparisons: Optional[int] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        require_repeatability_evals()
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.max_comparisons = max_comparisons
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky

    # ---- measuring ---------------------------------------------------------------------
    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case, multimodal=False)
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                    )
                )
            else:
                runs = get_runs(test_case.metadata, "runs")
                pairs = sample_pairs(len(runs), self.max_comparisons)
                judged = [
                    (i, j, self._judge(test_case.input, runs[i], runs[j]))
                    for i, j in pairs
                ]
                self._finalize(judged)
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        prepare_measure(self, test_case, multimodal=False)
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            runs = get_runs(test_case.metadata, "runs")
            pairs = sample_pairs(len(runs), self.max_comparisons)
            verdicts = await asyncio.gather(
                *(
                    self._a_judge(test_case.input, runs[i], runs[j])
                    for i, j in pairs
                )
            )
            self._finalize([(i, j, v) for (i, j), v in zip(pairs, verdicts)])
            return self.score

    # ---- judging -----------------------------------------------------------------------
    def _judge(self, input: str, a: str, b: str) -> AgreementVerdict:
        if _same(a, b):
            return _IDENTICAL
        return generate_with_schema_and_extract(
            metric=self,
            prompt=AnswerConsistencyTemplate.generate_verdict(input, a, b),
            schema_cls=AgreementVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: AgreementVerdict(**data),
        )

    async def _a_judge(self, input: str, a: str, b: str) -> AgreementVerdict:
        if _same(a, b):
            return _IDENTICAL
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=AnswerConsistencyTemplate.generate_verdict(input, a, b),
            schema_cls=AgreementVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: AgreementVerdict(**data),
        )

    # ---- scoring -----------------------------------------------------------------------
    def _finalize(self, judged: List[_Judged]) -> None:
        lib = require_repeatability_evals()
        weights = {d.name: d.weight for d in lib.DEFAULT_DIMENSIONS}
        total = sum(weights.values())
        pair_scores: List[float] = []
        per_dim: Dict[str, List[float]] = {k: [] for k in weights}
        for _, _, verdict in judged:
            scores = verdict.scores.model_dump()
            norm = {k: (min(max(scores[k], 1), 5) - 1) / 4 for k in weights}
            for k, v in norm.items():
                per_dim[k].append(v)
            pair_scores.append(
                sum(weights[k] * norm[k] for k in weights) / total
            )

        score = sum(pair_scores) / len(pair_scores)
        self.score_breakdown = {k: sum(v) / len(v) for k, v in per_dim.items()}
        self.score = 0 if self.strict_mode and score < self.threshold else score
        worst = min(range(len(judged)), key=lambda n: pair_scores[n])
        wi, wj, wv = judged[worst]
        weakest = min(
            self.score_breakdown, key=lambda k: self.score_breakdown[k]
        )
        if not self.include_reason:
            self.reason = None
        elif score >= 1.0:
            self.reason = (
                f"All {len(judged)} compared pairs of runs fully agree."
            )
        else:
            self.reason = (
                f"Compared {len(judged)} pairs of runs; mean agreement is "
                f"{score:.2f}. Least consistent dimension: {weakest} "
                f"({self.score_breakdown[weakest]:.2f}). Most different pair: "
                f"{describe_pair(wi, wj)} ({pair_scores[worst]:.2f})"
                + (f": {wv.reason}" if wv.reason else ".")
            )
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Pair scores: {[round(s, 2) for s in pair_scores]}",
                f"Per-dimension means: {self.score_breakdown}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

    @property
    def __name__(self):
        return "Answer Consistency"
