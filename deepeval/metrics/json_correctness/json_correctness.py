from typing import List, Union
import json
from pydantic import BaseModel, ValidationError

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
)
from deepeval.metrics.indicator import metric_progress_indicator


required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]


class JsonCorrectnessMetric(BaseMetric):
    def __init__(
        self,
        expected_schema: BaseModel,
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.expected_schema = expected_schema

    def measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[0]
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            valid_json = True
            try:
                self.expected_schema.model_validate_json(
                    test_case.actual_output
                )
            except ValidationError as e:
                valid_json = False
                if self.include_reason:
                    self.reason = self.generate_friendly_error_message(e)

            self.score = 1 if valid_json else 0
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"LLM outputed Json:\n{test_case.actual_output}",
                    f"Expected Json Schema:\n{json.dumps(self.expected_schema.model_json_schema(), indent=4)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        return self.measure(test_case, _show_indicator=_show_indicator)

    def generate_friendly_error_message(self, error: ValidationError) -> str:
        error_messages = []
        for err in error.errors():
            # Extract error location, message, and type
            loc = " -> ".join(map(str, err.get("loc", [])))
            msg = err.get("msg", "Unknown error")
            error_type = err.get("type", "Unknown type")

            # Format each error message in a readable way
            error_message = f"Error in '{loc}': {msg} (Type: {error_type})"
            error_messages.append(error_message)

        # Join all error messages into a single formatted string
        return "\n".join(error_messages)

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
