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
from deepeval.metrics.citation_faithfulness.template import (
    CitationFaithfulnessTemplate,
)
from deepeval.metrics.citation_faithfulness.schema import (
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
        threshold: float = 1.0,
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
        prompt = self._build_prompt(test_case)
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=CitationFaithfulnessVerdict,
            extract_schema=lambda s: s,
            extract_json=lambda data: CitationFaithfulnessVerdict(**data),
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

    def _calculate_score(self) -> float:
        faithful = self.verdict.verdict.strip().lower() == "faithful"
        score = 1.0 if faithful else 0.0
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
        return "Citation Faithfulness"
