from typing import List, Optional, Union
import asyncio

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import BaseMetric
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.community.named_reference_attribution.template import (
    NamedReferenceAttributionTemplate,
)
from deepeval.metrics.community.named_reference_attribution.schema import (
    NamedReference,
    NamedReferences,
    NamedReferenceVerdict,
    Verdicts,
    NamedReferenceAttributionScoreReason,
)


class NamedReferenceAttributionMetric(BaseMetric):
    """Named structural-reference attribution.

    Checks whether references in `actual_output` to a document's own
    structural labels (e.g. "Table 3", "Section 4.2", "footnote 4") are
    attributed to the right label, not just whether the claim is true
    somewhere in `retrieval_context`.

    This is a sibling to `CitationFaithfulnessMetric`, which checks `[N]`
    markers numbered by the metric itself. `NamedReferenceAttributionMetric`
    instead checks a document's own native labels, which a model may cite by
    name (e.g. "According to Table 3...") without ever emitting a `[N]`
    marker.

    The score is the fraction of named references whose claim is supported
    by the content that actually appears under that label in
    `retrieval_context`.
    """

    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
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
                self.references = self._extract_references(
                    test_case.actual_output
                )
                self.verdicts = self._generate_verdicts(
                    test_case.retrieval_context
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"References:\n{prettify_list(self.references)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
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
            self.references = await self._a_extract_references(
                test_case.actual_output
            )
            self.verdicts = await self._a_generate_verdicts(
                test_case.retrieval_context
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"References:\n{prettify_list(self.references)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    def _extract_references(self, actual_output: str) -> List[NamedReference]:
        prompt = NamedReferenceAttributionTemplate.extract_references(
            actual_output
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=NamedReferences,
            extract_schema=lambda s: list(s.references),
            extract_json=lambda data: [
                NamedReference(**item) for item in data["references"]
            ],
        )

    async def _a_extract_references(
        self, actual_output: str
    ) -> List[NamedReference]:
        prompt = NamedReferenceAttributionTemplate.extract_references(
            actual_output
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=NamedReferences,
            extract_schema=lambda s: list(s.references),
            extract_json=lambda data: [
                NamedReference(**item) for item in data["references"]
            ],
        )

    def _generate_verdicts(
        self, retrieval_context: List[str]
    ) -> List[NamedReferenceVerdict]:
        if len(self.references) == 0:
            return []

        prompt = NamedReferenceAttributionTemplate.generate_verdicts(
            references=self.references, retrieval_context=retrieval_context
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                NamedReferenceVerdict(**item) for item in data["verdicts"]
            ],
        )

    async def _a_generate_verdicts(
        self, retrieval_context: List[str]
    ) -> List[NamedReferenceVerdict]:
        if len(self.references) == 0:
            return []

        prompt = NamedReferenceAttributionTemplate.generate_verdicts(
            references=self.references, retrieval_context=retrieval_context
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                NamedReferenceVerdict(**item) for item in data["verdicts"]
            ],
        )

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        correct_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                correct_count += 1

        score = correct_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def _misattributions(self) -> List[str]:
        return [
            f'"{verdict.label}": {verdict.reason}'
            for verdict in self.verdicts
            if verdict.verdict.strip().lower() != "yes"
        ]

    def _generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        prompt = NamedReferenceAttributionTemplate.generate_reason(
            misattributions=self._misattributions(),
            score=format(self.score, ".2f"),
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=NamedReferenceAttributionScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_reason(self) -> Optional[str]:
        if self.include_reason is False:
            return None

        prompt = NamedReferenceAttributionTemplate.generate_reason(
            misattributions=self._misattributions(),
            score=format(self.score, ".2f"),
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=NamedReferenceAttributionScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    @property
    def __name__(self):
        return "Named Reference Attribution"
