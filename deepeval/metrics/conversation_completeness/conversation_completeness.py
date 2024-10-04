import asyncio
from typing import Optional, Union, Dict, List

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.conversation_completeness.template import (
    ConversationCompletenessTemplate,
)
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
from deepeval.metrics.conversation_completeness.schema import *

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]


class ConversationCompletenessMetric(BaseConversationalMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        window_size: int = 3,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.window_size = window_size

    def measure(
        self, test_case: ConversationalTestCase, _show_indicator: bool = True
    ):
        check_conversational_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.turns = format_turns(test_case.turns, required_params)
                self.user_intentions = self._extract_user_intentions()
                self.verdicts = [
                    self._generate_verdict(user_intention)
                    for user_intention in self.user_intentions
                ]

                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Turns:\n{prettify_list(self.turns)}",
                        f"User Intentions:\n{prettify_list(self.user_intentions)}",
                        f"Verdicts:\n{prettify_list(self.verdicts)}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
                return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
    ) -> float:
        check_conversational_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            self.turns = format_turns(test_case.turns, required_params)
            self.user_intentions = await self._a_extract_user_intentions()
            self.verdicts = await asyncio.gather(
                *[
                    self._a_generate_verdict(user_intention)
                    for user_intention in self.user_intentions
                ]
            )

            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Turns:\n{prettify_list(self.turns)}",
                    f"User Intentions:\n{prettify_list(self.user_intentions)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self) -> str:
        incompletenesses: List[str] = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                incompletenesses.append(verdict.reason)

        prompt = ConversationCompletenessTemplate.generate_reason(
            score=self.score,
            incompletenesses=incompletenesses,
            intentions=self.user_intentions,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        incompletenesses: List[str] = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                incompletenesses.append(verdict.reason)

        prompt = ConversationCompletenessTemplate.generate_reason(
            score=self.score,
            incompletenesses=incompletenesses,
            intentions=self.user_intentions,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdict(
        self, intention: str
    ) -> ConversationCompletenessVerdict:
        prompt = ConversationCompletenessTemplate.generate_verdicts(
            messages=self.turns, intention=intention
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return ConversationCompletenessVerdict(**data)
        else:
            try:
                res: ConversationCompletenessVerdict = (
                    await self.model.a_generate(
                        prompt, schema=ConversationCompletenessVerdict
                    )
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ConversationCompletenessVerdict(**data)

    def _generate_verdict(
        self, intention: str
    ) -> ConversationCompletenessVerdict:
        prompt = ConversationCompletenessTemplate.generate_verdicts(
            messages=self.turns, intention=intention
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return ConversationCompletenessVerdict(**data)
        else:
            try:
                res: ConversationCompletenessVerdict = self.model.generate(
                    prompt, schema=ConversationCompletenessVerdict
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ConversationCompletenessVerdict(**data)

    async def _a_extract_user_intentions(self) -> List[str]:
        prompt = ConversationCompletenessTemplate.extract_user_intentions(
            messages=self.turns
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return UserIntentions(**data).intentions
        else:
            try:
                res: UserIntentions = await self.model.a_generate(
                    prompt, schema=UserIntentions
                )
                return res.intentions
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return UserIntentions(**data).intentions

    def _extract_user_intentions(self) -> List[str]:
        prompt = ConversationCompletenessTemplate.extract_user_intentions(
            messages=self.turns
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return UserIntentions(**data).intentions
        else:
            try:
                res: UserIntentions = self.model.generate(
                    prompt, schema=UserIntentions
                )
                return res.intentions
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return UserIntentions(**data).intentions

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() != "no":
                relevant_count += 1

        score = relevant_count / number_of_verdicts
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
        return "Conversation Completeness"
