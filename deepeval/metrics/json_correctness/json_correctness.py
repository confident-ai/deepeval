from typing import List, Optional, Union
import json
from pydantic import BaseModel, ValidationError

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
)
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    trimAndLoadJson,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.json_correctness.template import JsonCorrectnessTemplate
from deepeval.metrics.json_correctness.schema import JsonCorrectnessScoreReason
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.api import metric_data_manager

DEFAULT_CORRECT_REASON = "The generated Json matches and is syntactically correct to the expected schema."


class JsonCorrectnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]

    def __init__(
        self,
        expected_schema: BaseModel,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        async_mode: bool = True,
        include_reason: bool = True,
        strict_mode: bool = True,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.expected_schema = expected_schema

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        check_llm_test_case_params(test_case, self._required_params, self)

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
                valid_json = True
                try:
                    self.expected_schema.model_validate_json(
                        test_case.actual_output
                    )
                except ValidationError as e:
                    valid_json = False

                self.score = 1 if valid_json else 0
                self.reason = self.generate_reason(test_case.actual_output)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"LLM outputed Json:\n{test_case.actual_output}",
                        # f"Expected Json Schema:\n{json.dumps(self.expected_schema.model_json_schema(), indent=4)}",
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

        check_llm_test_case_params(test_case, self._required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            valid_json = True
            try:
                self.expected_schema.model_validate_json(
                    test_case.actual_output
                )
            except ValidationError as e:
                valid_json = False

            self.score = 1 if valid_json else 0
            self.reason = await self.a_generate_reason(test_case.actual_output)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"LLM outputed Json:\n{test_case.actual_output}",
                    # f"Expected Json Schema:\n{json.dumps(self.expected_schema.model_json_schema(), indent=4)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def a_generate_reason(self, actual_output: str) -> str:
        if self.include_reason is False:
            return None

        is_valid_json = self.score == 1
        if is_valid_json:
            return DEFAULT_CORRECT_REASON

        prompt: dict = JsonCorrectnessTemplate.generate_reason(
            actual_output=actual_output,
            expected_schema=json.dumps(
                self.expected_schema.model_json_schema(), indent=4
            ),
            is_valid_json=is_valid_json,
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=JsonCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: JsonCorrectnessScoreReason = await self.model.a_generate(
                    prompt, schema=JsonCorrectnessScoreReason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def generate_reason(self, actual_output: str) -> str:
        if self.include_reason is False:
            return None

        is_valid_json = self.score == 1
        if is_valid_json:
            return DEFAULT_CORRECT_REASON

        prompt: dict = JsonCorrectnessTemplate.generate_reason(
            actual_output=actual_output,
            expected_schema=json.dumps(
                self.expected_schema.model_json_schema(), indent=4
            ),
            is_valid_json=is_valid_json,
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=JsonCorrectnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: JsonCorrectnessScoreReason = self.model.generate(
                    prompt, schema=JsonCorrectnessScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Json Correctness"
