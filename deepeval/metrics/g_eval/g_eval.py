"""LLM evaluated metric based on the GEval framework: https://arxiv.org/pdf/2303.16634.pdf"""

from typing import Optional, List, Tuple, Union
from pydantic import BaseModel

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval.template import GEvalTemplate
from deepeval.utils import (
    trimAndLoadJson,
    check_test_case_params,
    get_or_create_event_loop,
)
from deepeval.models import GPTModel, DeepEvalBaseLLM
from deepeval.telemetry import capture_metric_type
from deepeval.metrics.indicator import metric_progress_indicator

G_EVAL_PARAMS = {
    LLMTestCaseParams.INPUT: "Input",
    LLMTestCaseParams.ACTUAL_OUTPUT: "Actual Output",
    LLMTestCaseParams.EXPECTED_OUTPUT: "Expected Output",
    LLMTestCaseParams.CONTEXT: "Context",
    LLMTestCaseParams.RETRIEVAL_CONTEXT: "Retrieval Context",
}


def construct_g_eval_params_string(
    llm_test_case_params: List[LLMTestCaseParams],
):
    g_eval_params = [G_EVAL_PARAMS[param] for param in llm_test_case_params]

    if len(g_eval_params) == 1:
        g_eval_params_str = g_eval_params[0]
    elif len(g_eval_params) == 2:
        g_eval_params_str = " and ".join(g_eval_params)
    else:
        g_eval_params_str = (
            ", ".join(g_eval_params[:-1]) + ", and " + g_eval_params[-1]
        )

    return g_eval_params_str


class GEvalResponse(BaseModel):
    score: float
    reason: str


class GEval(BaseMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: List[LLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.name = name
        self.evaluation_params = evaluation_params

        # Check if both criteria and evaluation_steps are not None at the same time
        if criteria is None and evaluation_steps is None:
            raise ValueError(
                "Either 'criteria' or 'evaluation_steps' must be provided."
            )

        # Check if criteria is provided, it cannot be an empty string
        if criteria is not None and not criteria.strip():
            raise ValueError("Criteria provided cannot be an empty string.")

        # Check if evaluation_steps is provided, it cannot be an empty list
        if evaluation_steps is not None and len(evaluation_steps) == 0:
            raise ValueError(
                "'evaluation_steps' must not be an empty list. Either omit evaluation steps or include a non-empty list of steps."
            )

        self.criteria = criteria
        if isinstance(model, DeepEvalBaseLLM):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()
        self.evaluation_steps = evaluation_steps
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode

    def measure(self, test_case: LLMTestCase) -> float:
        check_test_case_params(
            test_case, self.evaluation_params, f"GEval({self.__name__})"
        )

        with metric_progress_indicator(self):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.evaluation_steps: List[str] = (
                    self._generate_evaluation_steps()
                )
                g_score, reason = self.evaluate(test_case)
                self.reason = reason
                self.score = float(g_score) / 10
                self.score = (
                    0
                    if self.strict_mode and self.score < self.threshold
                    else self.score
                )
                self.success = self.score >= self.threshold
                capture_metric_type(self.__name__)
                return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = True
    ) -> float:
        check_test_case_params(
            test_case, self.evaluation_params, f"GEval({self.__name__})"
        )

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.evaluation_steps: List[str] = (
                await self._a_generate_evaluation_steps()
            )
            g_score, reason = await self._a_evaluate(test_case)
            self.reason = reason
            self.score = float(g_score) / 10
            self.score = (
                0
                if self.strict_mode and self.score < self.threshold
                else self.score
            )
            self.success = self.score >= self.threshold
            capture_metric_type(self.__name__)
            return self.score

    async def _a_generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = GEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res)
        return data["steps"]

    def _generate_evaluation_steps(self) -> List[str]:
        if self.evaluation_steps:
            return self.evaluation_steps

        g_eval_params_str = construct_g_eval_params_string(
            self.evaluation_params
        )
        prompt = GEvalTemplate.generate_evaluation_steps(
            criteria=self.criteria, parameters=g_eval_params_str
        )
        res = self.model.generate(prompt)
        data = trimAndLoadJson(res)
        return data["steps"]

    async def _a_evaluate(self, test_case: LLMTestCase) -> Tuple[int, str]:
        text = """"""
        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{G_EVAL_PARAMS[param]}:\n{value} \n\n"

        prompt = GEvalTemplate.generate_evaluation_results(
            evaluation_steps=self.number_evaluation_steps(),
            text=text,
        )
        res = await self.model.a_generate(prompt)
        data = trimAndLoadJson(res)
        return data["score"], data["reason"]

    def evaluate(self, test_case: LLMTestCase) -> Tuple[int, str]:
        text = """"""
        for param in self.evaluation_params:
            value = getattr(test_case, param.value)
            text += f"{param.value}: {value} \n\n"

        prompt = GEvalTemplate.generate_evaluation_results(
            evaluation_steps=self.number_evaluation_steps(),
            text=text,
        )
        res = self.model.generate(prompt)
        data = trimAndLoadJson(res)
        return data["score"], data["reason"]

    def number_evaluation_steps(self):
        evaluation_steps = """"""
        for index, string in enumerate(self.evaluation_steps, start=1):
            evaluation_steps += f"{index}. {string}\n"
        return evaluation_steps

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return f"GEval ({self.name})"
