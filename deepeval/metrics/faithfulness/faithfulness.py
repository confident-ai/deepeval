from typing import List, Optional, Union, Type
import asyncio

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.faithfulness.schema import (
    FaithfulnessVerdict,
    Verdicts,
    FaithfulnessScoreReason,
    Truths,
    Claims,
)
from deepeval.metrics.api import metric_data_manager


class FaithfulnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        truths_extraction_limit: Optional[int] = None,
        penalize_ambiguous_claims: bool = False,
        evaluation_template: Type[FaithfulnessTemplate] = FaithfulnessTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
        self.penalize_ambiguous_claims = penalize_ambiguous_claims

        self.truths_extraction_limit = truths_extraction_limit
        if self.truths_extraction_limit is not None:
            self.truths_extraction_limit = max(self.truths_extraction_limit, 0)

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
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
                retrieval_context = test_case.retrieval_context
                actual_output = test_case.actual_output

                self.truths = self._generate_truths(
                    retrieval_context, multimodal
                )
                self.claims = self._generate_claims(actual_output, multimodal)
                self.verdicts = self._generate_verdicts(multimodal)
                self.score = self._calculate_score()
                self.reason = self._generate_reason(multimodal)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                        f"Claims:\n{prettify_list(self.claims)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            retrieval_context = test_case.retrieval_context
            actual_output = test_case.actual_output

            self.truths, self.claims = await asyncio.gather(
                self._a_generate_truths(retrieval_context, multimodal),
                self._a_generate_claims(actual_output, multimodal),
            )
            self.verdicts = await self._a_generate_verdicts(multimodal)
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(multimodal)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths (limit={self.truths_extraction_limit}):\n{prettify_list(self.truths)}",
                    f"Claims:\n{prettify_list(self.claims)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def _a_generate_reason(self, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=FaithfulnessScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            contradictions=contradictions,
            score=format(self.score, ".2f"),
            multimodal=multimodal,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=FaithfulnessScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdicts(
        self, multimodal: bool
    ) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            claims=self.claims,
            retrieval_context="\n\n".join(self.truths),
            multimodal=multimodal,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                FaithfulnessVerdict(**item) for item in data["verdicts"]
            ],
        )

    def _generate_verdicts(self, multimodal: bool) -> List[FaithfulnessVerdict]:
        if len(self.claims) == 0:
            return []

        prompt = self.evaluation_template.generate_verdicts(
            claims=self.claims,
            retrieval_context="\n\n".join(self.truths),
            multimodal=multimodal,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: list(s.verdicts),
            extract_json=lambda data: [
                FaithfulnessVerdict(**item) for item in data["verdicts"]
            ],
        )

    async def _a_generate_truths(
        self, retrieval_context: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_truths(
            retrieval_context="\n\n".join(retrieval_context),
            extraction_limit=self.truths_extraction_limit,
            multimodal=multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Truths,
            extract_schema=lambda s: s.truths,
            extract_json=lambda data: data["truths"],
        )

    def _generate_truths(
        self, retrieval_context: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_truths(
            retrieval_context="\n\n".join(retrieval_context),
            extraction_limit=self.truths_extraction_limit,
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Truths,
            extract_schema=lambda s: s.truths,
            extract_json=lambda data: data["truths"],
        )

    async def _a_generate_claims(
        self, actual_output: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_claims(
            actual_output=actual_output, multimodal=multimodal
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Claims,
            extract_schema=lambda s: s.claims,
            extract_json=lambda data: data["claims"],
        )

    def _generate_claims(
        self, actual_output: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_claims(
            actual_output=actual_output, multimodal=multimodal
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Claims,
            extract_schema=lambda s: s.claims,
            extract_json=lambda data: data["claims"],
        )

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

            if (
                self.penalize_ambiguous_claims
                and verdict.verdict.strip().lower() == "idk"
            ):
                faithfulness_count -= 1

        score = faithfulness_count / number_of_verdicts
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
        return "Faithfulness"
