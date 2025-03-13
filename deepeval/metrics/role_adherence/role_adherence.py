from typing import Optional, Union, List, Dict

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.role_adherence.schema import (
    OutOfCharacterResponseVerdicts,
)
from deepeval.metrics.role_adherence.template import RoleAdherenceTemplate
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    format_turns,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import (
    LLMTestCaseParams,
    LLMTestCase,
    ConversationalTestCase,
)
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.role_adherence.schema import *


class RoleAdherenceMetric(BaseConversationalMetric):

    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
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
        self, test_case: ConversationalTestCase, _show_indicator: bool = True
    ):
        check_conversational_test_case_params(
            test_case, self._required_params, self, require_chatbot_role=True
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.turns: List[Dict[str, str]] = format_turns(
                    test_case.turns, self._required_params
                )
                self.out_of_character_verdicts: (
                    OutOfCharacterResponseVerdicts
                ) = self._extract_out_of_character_verdicts(
                    test_case.turns, test_case.chatbot_role
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(role=test_case.chatbot_role)
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Chatbot Role:\n{test_case.chatbot_role}",
                        f"Out-of-Character Turn Response(s):\n{prettify_list(self.out_of_character_verdicts.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
    ) -> float:
        check_conversational_test_case_params(
            test_case, self._required_params, self, require_chatbot_role=True
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            self.turns: List[Dict[str, str]] = format_turns(
                test_case.turns, self._required_params
            )
            self.out_of_character_verdicts = (
                await (
                    self._a_extract_out_of_character_verdicts(
                        test_case.turns, test_case.chatbot_role
                    )
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(
                role=test_case.chatbot_role
            )
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Chatbot Role:\n{test_case.chatbot_role}",
                    f"Out-of-Character Turn(s) Response(s):\n{prettify_list(self.out_of_character_verdicts.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, role: str) -> str:
        if self.include_reason is False:
            return None

        prompt = RoleAdherenceTemplate.generate_reason(
            score=self.score,
            role=role,
            out_of_character_responses=self.out_of_character_verdicts,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Reason)
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, role: str) -> str:
        prompt = RoleAdherenceTemplate.generate_reason(
            score=self.score,
            role=role,
            out_of_character_responses=self.out_of_character_verdicts.verdicts,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Reason)
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_extract_out_of_character_verdicts(
        self, llm_test_cases: List[LLMTestCase], role: str
    ) -> OutOfCharacterResponseVerdicts:
        prompt = (
            RoleAdherenceTemplate.extract_out_of_character_response_verdicts(
                turns=self.turns,
                role=role,
            )
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=OutOfCharacterResponseVerdicts
            )
            self.evaluation_cost += cost
        else:
            try:
                res: OutOfCharacterResponseVerdicts = (
                    await self.model.a_generate(
                        prompt, schema=OutOfCharacterResponseVerdicts
                    )
                )
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                res = OutOfCharacterResponseVerdicts(**data)

        for verdict in res.verdicts:
            try:
                index = verdict.index
                verdict.actual_output = (
                    f"{llm_test_cases[index].actual_output} (turn #{index+1})"
                )
            except:
                pass
        return res

    def _extract_out_of_character_verdicts(
        self, llm_test_cases: List[LLMTestCase], role: str
    ) -> OutOfCharacterResponseVerdicts:
        prompt = (
            RoleAdherenceTemplate.extract_out_of_character_response_verdicts(
                turns=self.turns,
                role=role,
            )
        )
        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=OutOfCharacterResponseVerdicts
            )
            self.evaluation_cost += cost
        else:
            try:
                res: OutOfCharacterResponseVerdicts = self.model.generate(
                    prompt, schema=OutOfCharacterResponseVerdicts
                )
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                res = OutOfCharacterResponseVerdicts(**data)

        for verdict in res.verdicts:
            try:
                index = verdict.index
                verdict.actual_output = (
                    f"{llm_test_cases[index].actual_output} (turn #{index+1})"
                )
            except:
                pass
        return res

    def _calculate_score(self) -> float:
        number_of_turns = len(self.turns)
        if number_of_turns == 0:
            return 1

        score = (
            number_of_turns
            - min(len(self.out_of_character_verdicts.verdicts), number_of_turns)
        ) / number_of_turns
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Role Adherence"
