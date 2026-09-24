from typing import List, Optional, Union

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.metrics.base_metric import Verdict
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    initialize_system_one_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneBinarySpec,
    SystemOneEvalSpec,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    system_one_probability,
    a_system_one_probability,
    format_decision_reason,
    verdict_from_probability,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.community.citation_faithfulness.template import (
    CitationFaithfulnessTemplate,
)
from deepeval.metrics.community.citation_faithfulness.schema import (
    CitationFaithfulnessVerdict,
)


class CitationFaithfulnessMetric(BaseMetric):
    """Citation-attribution faithfulness.

    Checks whether every ``[N]`` citation marker in ``actual_output`` points to
    the passage in ``retrieval_context`` that actually supports the specific
    claim the marker is attached to.

    This is stricter than ``FaithfulnessMetric``, which only checks whether a
    claim is supported by the retrieval context somewhere. ``CitationFaithfulness``
    additionally catches misattribution: a claim cited to passage ``[A]`` that
    does not support it, even when another passage ``[B]`` in the context would.

    The judge returns a binary verdict. A ``faithful`` verdict scores ``1.0`` and
    a ``unfaithful`` verdict scores ``0.0``. With the default threshold of
    ``1.0``, only a faithful answer is successful.

    The passages in ``retrieval_context`` are numbered ``[1]``, ``[2]``, ...
    before being shown to the judge, so the ``[N]`` markers in the answer resolve
    to the matching passage.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 1.0,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            False,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
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
                if run_system_one_eval(self, test_case):
                    return self._round_system_one_score()

                self.verdict = self._generate_verdict(test_case)
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Verdict:\n{self.verdict.verdict}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            False,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self._round_system_one_score()

            self.verdict = await self._a_generate_verdict(test_case)
            self.score = self._calculate_score()
            self.reason = self._generate_reason()
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdict:\n{self.verdict.verdict}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def _build_prompt(self, test_case: LLMTestCase) -> str:
        numbered_passages = CitationFaithfulnessTemplate.number_passages(
            test_case.retrieval_context
        )
        return CitationFaithfulnessTemplate.generate_verdict(
            input=test_case.input,
            numbered_passages=numbered_passages,
            actual_output=test_case.actual_output,
        )

    async def _a_generate_verdict(
        self, test_case: LLMTestCase
    ) -> CitationFaithfulnessVerdict:
        p = await a_system_one_probability(
            self, self._experimental_system_one_spec(test_case)
        )
        if p is not None:
            return self._system_one_verdict(p)
        prompt = self._build_prompt(test_case)
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=CitationFaithfulnessVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: CitationFaithfulnessVerdict(**data),
        )

    def _generate_verdict(
        self, test_case: LLMTestCase
    ) -> CitationFaithfulnessVerdict:
        p = system_one_probability(
            self, self._experimental_system_one_spec(test_case)
        )
        if p is not None:
            return self._system_one_verdict(p)
        prompt = self._build_prompt(test_case)
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=CitationFaithfulnessVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: CitationFaithfulnessVerdict(**data),
        )

    def _experimental_system_one_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneBinarySpec]:
        if test_case.multimodal:
            return None
        return SystemOneBinarySpec(
            instructions=CitationFaithfulnessTemplate._experimental_system_one_verdict(),
            state={
                "input": test_case.input,
                "passages": CitationFaithfulnessTemplate.number_passage_list(
                    test_case.retrieval_context
                ),
                "actual_output": test_case.actual_output,
            },
        )

    def _system_one_verdict(self, p: float) -> CitationFaithfulnessVerdict:
        return CitationFaithfulnessVerdict(
            verdict=(
                "faithful"
                if verdict_from_probability(p) == Verdict.YES
                else "unfaithful"
            ),
            reasoning=format_decision_reason(self, "P(faithful)", p),
        )

    def _system_one_eval_spec(
        self, test_case: LLMTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole metric as one Jev request over
        `input`, `actual_output` and the numbered `passages` of
        `retrieval_context`; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            questions=parse_questions(
                CitationFaithfulnessTemplate._experimental_system_one_questions()
            ),
            extra_state={
                "passages": CitationFaithfulnessTemplate.number_passage_list(
                    test_case.retrieval_context
                )
            },
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None
        if self.verdict.reasoning:
            return self.verdict.reasoning
        return (
            "Every citation marker points to a passage that supports its claim."
            if self.verdict.verdict.strip().lower() == "faithful"
            else "At least one citation marker points to a passage that does not support its claim."
        )

    def _round_system_one_score(self) -> float:
        """Jev's score is a weighted mean; this metric is pass/fail, so it
        is rounded onto the same 1.0 / 0.0 the LLM verdict gives."""
        raw = self.score
        self.score = 1.0 if raw >= 0.5 else 0.0
        if self.strict_mode and self.score < self.threshold:
            self.score = 0
        if self.reason is not None:
            self.reason += (
                f"\nRounded {raw:.2f} to {self.score:g}: citations are "
                f"either faithful or not."
            )
        self.success = self.is_successful()
        return self.score

    def _calculate_score(self) -> float:
        faithful = self.verdict.verdict.strip().lower() == "faithful"
        score = 1.0 if faithful else 0.0
        return 0 if self.strict_mode and score < self.threshold else score

    @property
    def __name__(self):
        return "Citation Faithfulness"
