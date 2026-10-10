import re
from itertools import combinations
from typing import Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

_SAMPLE_KEYS = ("samples", "completions", "sampled_outputs")


def _tokens(text: str) -> List[str]:
    return re.sub(r"\s+", " ", text.strip().lower()).split()


def _jaccard(first: List[str], second: List[str]) -> float:
    first_set, second_set = set(first), set(second)
    union = first_set | second_set
    if not union:
        return 1.0
    return len(first_set & second_set) / len(union)


class SelfConsistencyMetric(BaseMetric):
    """Do sampled answers to the same input agree with each other?

    Scores the mean pairwise token agreement across ``actual_output``
    plus the extra samples passed via ``samples`` (or
    ``test_case.metadata`` under ``samples``, ``completions`` or
    ``sampled_outputs``). ``1.0`` means every sample says the same
    thing; lower values mean the model is unstable on this input.
    Agreement is lexical (unigram Jaccard), so the metric is fully
    **deterministic**: no LLM, no API key, zero token cost. For
    semantic agreement, use ``GEval`` with a consistency criterion.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        # Deterministic metric: no evaluation model is used.
        self.model = None
        self.using_native_model = False
        self.async_mode = False

    def measure(
        self,
        test_case: LLMTestCase,
        samples: Optional[List[str]] = None,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            None,
            test_case.multimodal,
        )
        self.test_case = test_case
        with metric_progress_indicator(
            self,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            pool = [test_case.actual_output] + self._samples(test_case, samples)
            if len(pool) < 2:
                raise ValueError(
                    "SelfConsistencyMetric needs at least 2 outputs: "
                    "`actual_output` plus `samples` (or metadata)."
                )
            pairs = [
                _jaccard(_tokens(first), _tokens(second))
                for first, second in combinations(pool, 2)
            ]
            mean_pairwise = sum(pairs) / len(pairs)
            self.score_breakdown = {
                "self_consistency": mean_pairwise,
                "mean_pairwise": mean_pairwise,
                "min_pairwise": min(pairs),
                "n_samples": float(len(pool)),
            }
            self.score = (
                0
                if self.strict_mode and mean_pairwise < self.threshold
                else mean_pairwise
            )
            self.success = self.is_successful()
            self.reason = self._generate_reason(mean_pairwise, len(pool))
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Samples: {len(pool)}",
                    f"Mean pairwise agreement: {mean_pairwise:.4f}",
                    f"Min pairwise agreement: {min(pairs):.4f}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        samples: Optional[List[str]] = None,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        # Deterministic metric — no async work to do; reuse the sync path.
        return self.measure(
            test_case,
            samples=samples,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    @staticmethod
    def _samples(
        test_case: LLMTestCase, samples: Optional[List[str]]
    ) -> List[str]:
        if samples is not None:
            return list(samples)
        metadata: Dict = test_case.metadata or {}
        for key in _SAMPLE_KEYS:
            found = metadata.get(key)
            if isinstance(found, list) and all(
                isinstance(item, str) for item in found
            ):
                return list(found)
        return []

    def _generate_reason(self, agreement: float, count: int) -> Optional[str]:
        if not self.include_reason:
            return None
        return (
            f"{count} sampled answer(s) agree at {agreement:.2f} "
            "mean pairwise token overlap."
        )

    @property
    def __name__(self):
        return "Output Self-Consistency"
