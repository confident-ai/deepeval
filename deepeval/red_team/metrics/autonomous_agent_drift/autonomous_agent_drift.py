from typing import Optional, List, Tuple, Union


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
from deepteam.metrics.types import EvaluationExample
from deepteam.metrics.model_reply_screening import (
    single_turn_reply_bypasses_llm_judge,
    DECLINE_OR_EMPTY_PASS_REASON,
)

from deepteam.metrics.autonomous_agent_drift.schema import *
from deepteam.metrics.autonomous_agent_drift.template import (
    AutonomousAgentDriftTemplate,
)
from deepteam.utils import format_turns, format_tools_called


class AutonomousAgentDriftMetric(BaseRedTeamingMetric):
    def __init__(
        self,
        drift_type: str,
        purpose: Optional[
            str
        ] = None,  # Added: application context for evaluation
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_examples: Optional[List[EvaluationExample]] = None,
        evaluation_guidelines: Optional[List[str]] = None,
    ):
        self.drift_type = drift_type
        self.purpose = purpose  # Added: store purpose for template
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_examples = evaluation_examples
        self.evaluation_guidelines = evaluation_guidelines

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
                    steps=[
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score

    async def a_measure(
        self,
        test_case: RTTestCase,
        _show_indicator: bool = False,
    ) -> float:

        self.evaluation_cost = None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            score, reason = await self._a_evaluate(test_case)
            self.reason = reason
            self.score = score
            self.success = self.score == 1
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

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

        all_tools_executed = []
        if test_case.tools_called:
            all_tools_executed.extend(test_case.tools_called)
        if test_case.turns:
            for turn in test_case.turns:
                if turn.tools_called:
                    all_tools_executed.extend(turn.tools_called)

        tools_called = None
        if all_tools_executed:
            tools_called = format_tools_called(all_tools_executed)

        all_retrieval_contexts = []
        if test_case.retrieval_context:
            all_retrieval_contexts.extend(test_case.retrieval_context)
        if test_case.turns:
            for turn in test_case.turns:
                if turn.retrieval_context:
                    all_retrieval_contexts.extend(turn.retrieval_context)

        retrieval_context = (
            all_retrieval_contexts if all_retrieval_contexts else None
        )

        vulnerability_type_label = format_vulnerability_type_label(
            test_case.vulnerability_type
        )
        prompt = AutonomousAgentDriftTemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=actual_output,
            drift_type=self.drift_type,
            purpose=self.purpose,  # Added: pass purpose to template
            retrieval_context=retrieval_context,
            tools_called=tools_called,
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

        all_tools_executed = []
        if test_case.tools_called:
            all_tools_executed.extend(test_case.tools_called)
        if test_case.turns:
            for turn in test_case.turns:
                if turn.tools_called:
                    all_tools_executed.extend(turn.tools_called)

        tools_called = None
        if all_tools_executed:
            tools_called = format_tools_called(all_tools_executed)

        all_retrieval_contexts = []
        if test_case.retrieval_context:
            all_retrieval_contexts.extend(test_case.retrieval_context)
        if test_case.turns:
            for turn in test_case.turns:
                if turn.retrieval_context:
                    all_retrieval_contexts.extend(turn.retrieval_context)

        retrieval_context = (
            all_retrieval_contexts if all_retrieval_contexts else None
        )

        vulnerability_type_label = format_vulnerability_type_label(
            test_case.vulnerability_type
        )
        prompt = AutonomousAgentDriftTemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=actual_output,
            drift_type=self.drift_type,
            purpose=self.purpose,  # Added: pass purpose to template
            retrieval_context=retrieval_context,
            tools_called=tools_called,
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
        return f"Autonomous Agent Drift (Red Teaming)"
