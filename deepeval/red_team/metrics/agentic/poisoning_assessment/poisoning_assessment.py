from typing import List, Optional, Tuple, Union

from deepteam.test_case import RTTestCase
from deepteam.metrics.evaluation_prompt_blocks import (
    format_vulnerability_type_label,
)
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator

from deepteam.metrics import BaseRedTeamingMetric
from deepteam.metrics.agentic.poisoning_assessment.template import (
    PoisoningAssessmentTemplate,
)
from deepteam.metrics.agentic.poisoning_assessment.schema import *
from deepteam.utils import format_turns
from deepteam.metrics.types import EvaluationExample
from deepteam.metrics.model_reply_screening import (
    single_turn_reply_bypasses_llm_judge,
    DECLINE_OR_EMPTY_PASS_REASON,
)


class PoisoningAssessmentMetric(BaseRedTeamingMetric):
    def __init__(
        self,
        purpose: str,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_examples: Optional[List[EvaluationExample]] = None,
        evaluation_guidelines: Optional[List[str]] = None,
    ):
        self.purpose = purpose
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_examples = evaluation_examples
        self.evaluation_guidelines = evaluation_guidelines
        try:
            self.system_prompt = model.get_system_prompt()
        except:
            self.system_prompt = ""

    def measure(self, test_case: RTTestCase) -> float:

        self.evaluation_cost = None
        with metric_progress_indicator(self, _show_indicator=False):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                score, reason = self.evaluate(test_case)
                self.reason = reason
                self.score = score
                self.success = self.score == 1
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[f"Score: {self.score}\nReason: {self.reason}"],
                )
                return self.score

    async def a_measure(
        self,
        test_case: RTTestCase,
        _show_indicator: bool = False,
    ) -> float:

        self.evaluation_cost = None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            score, reason = await self._a_evaluate(test_case)
            self.reason = reason
            self.score = score
            self.success = self.score == 1
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[f"Score: {self.score}\nReason: {self.reason}"],
            )
            return self.score

    async def a_generate_purpose(self):
        if self.purpose:
            return self.purpose
        prompt = PoisoningAssessmentTemplate.extract_purpose(self.system_prompt)
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt=prompt, schema=Purpose
            )
            if cost is not None:
                self.evaluation_cost = (self.evaluation_cost or 0) + cost
            return res.purpose
        else:
            try:
                res: Purpose = await self.model.a_generate(
                    prompt=prompt, schema=Purpose
                )
                return res.purpose
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["purpose"]

    def generate_purpose(self):
        if self.purpose:
            return self.purpose
        prompt = PoisoningAssessmentTemplate.extract_purpose(self.system_prompt)
        if self.using_native_model:
            res, cost = self.model.generate(prompt=prompt, schema=Purpose)
            if cost is not None:
                self.evaluation_cost = (self.evaluation_cost or 0) + cost
            return res.purpose
        else:
            try:
                res: Purpose = self.model.generate(
                    prompt=prompt, schema=Purpose
                )
                return res.purpose
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["purpose"]

    async def _a_evaluate(
        self, test_case: RTTestCase
    ) -> Tuple[Union[int, float], str]:

        if test_case.turns is not None:
            actual_output = format_turns(test_case.turns)
        else:
            if single_turn_reply_bypasses_llm_judge(test_case.actual_output):
                return 1, DECLINE_OR_EMPTY_PASS_REASON
            actual_output = (
                "AI'S ACTUAL OUTPUT TO EVALUATE: \n"
                + test_case.actual_output
                + "\n"
            )

        vulnerability_type_label = format_vulnerability_type_label(
            test_case.vulnerability_type
        )
        prompt = PoisoningAssessmentTemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=actual_output,
            purpose=self.purpose,
            evaluation_examples=self.evaluation_examples,
            evaluation_guidelines=self.evaluation_guidelines,
            vulnerability_type_label=vulnerability_type_label,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt=prompt, schema=ReasonScore
            )
            if cost is not None:
                self.evaluation_cost = (self.evaluation_cost or 0) + cost
            return res.score, res.reason
        else:
            try:
                res: ReasonScore = await self.model.a_generate(
                    prompt=prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]

    def evaluate(self, test_case: RTTestCase) -> Tuple[Union[int, float], str]:

        if test_case.turns is not None:
            actual_output = format_turns(test_case.turns)
        else:
            if single_turn_reply_bypasses_llm_judge(test_case.actual_output):
                return 1, DECLINE_OR_EMPTY_PASS_REASON
            actual_output = (
                "AI'S ACTUAL OUTPUT TO EVALUATE: \n"
                + test_case.actual_output
                + "\n"
            )

        vulnerability_type_label = format_vulnerability_type_label(
            test_case.vulnerability_type
        )
        prompt = PoisoningAssessmentTemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=actual_output,
            purpose=self.purpose,
            evaluation_examples=self.evaluation_examples,
            evaluation_guidelines=self.evaluation_guidelines,
            vulnerability_type_label=vulnerability_type_label,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt=prompt, schema=ReasonScore)
            if cost is not None:
                self.evaluation_cost = (self.evaluation_cost or 0) + cost
            return res.score, res.reason
        else:
            try:
                res: ReasonScore = self.model.generate(
                    prompt=prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score == 1
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return f"Poisoning Assessment (Red Teaming)"
