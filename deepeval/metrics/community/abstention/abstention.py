from typing import List, Optional, Union

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.community.abstention.template import AbstentionTemplate
from deepeval.metrics.community.abstention.schema import AbstentionVerdict


class AbstentionMetric(BaseMetric):
    """Abstention (answer-vs-decline) correctness.

    Grades whether a retrieval-augmented answer made the right decision about
    *whether to answer at all*, given its ``retrieval_context``:

    - the context supports an answer and the system answered      -> correct
    - the context does not support one and the system abstained   -> correct
    - the context supports one but the system abstained anyway     -> over-refusal
    - the context does not support one but the system answered     -> unsupported answer

    Where ``FaithfulnessMetric`` and ``HallucinationMetric`` grade the *content*
    of an answer, ``AbstentionMetric`` grades the *decision to answer or
    decline*. It rewards a system that says "I don't know" exactly when it
    should, and penalises both over-refusal (declining an answerable question)
    and answering a question the context cannot support.

    A single judge makes two boolean determinations in one call — whether the
    context supports an answer, and whether the output abstained. The decision
    is correct when ``context_supports_answer`` equals ``not output_abstained``,
    scoring ``1.0``; otherwise ``0.0``. With the default threshold of ``0.5``,
    only a correct decision is successful.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
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
                        _log_metric_to_confident=_log_metric_to_confident,
                    )
                )
            else:
                self.verdict = self._generate_verdict(test_case)
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Context supports answer: {self.verdict.context_supports_answer}\n"
                        f"Output abstained: {self.verdict.output_abstained}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
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
            self.verdict = await self._a_generate_verdict(test_case)
            self.score = self._calculate_score()
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Context supports answer: {self.verdict.context_supports_answer}\n"
                    f"Output abstained: {self.verdict.output_abstained}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def _build_prompt(self, test_case: LLMTestCase) -> str:
        numbered_passages = AbstentionTemplate.number_passages(
            test_case.retrieval_context
        )
        return AbstentionTemplate.generate_verdict(
            input=test_case.input,
            numbered_passages=numbered_passages,
            actual_output=test_case.actual_output,
        )

    async def _a_generate_verdict(
        self, test_case: LLMTestCase
    ) -> AbstentionVerdict:
        prompt = self._build_prompt(test_case)
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AbstentionVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: AbstentionVerdict(**data),
        )

    def _generate_verdict(self, test_case: LLMTestCase) -> AbstentionVerdict:
        prompt = self._build_prompt(test_case)
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=AbstentionVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: AbstentionVerdict(**data),
        )

    def _decision_is_correct(self) -> bool:
        return self.verdict.context_supports_answer == (
            not self.verdict.output_abstained
        )

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None
        if self.verdict.reasoning:
            return self.verdict.reasoning
        supports = self.verdict.context_supports_answer
        abstained = self.verdict.output_abstained
        if supports and not abstained:
            return "The context supports an answer and the system answered — a correct decision."
        if not supports and abstained:
            return "The context does not support an answer and the system abstained — a correct decision."
        if supports and abstained:
            return "The context supports an answer but the system abstained — an over-refusal."
        return "The context does not support an answer but the system answered anyway — an unsupported answer."

    def _calculate_score(self) -> float:
        score = 1.0 if self._decision_is_correct() else 0.0
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Abstention"
