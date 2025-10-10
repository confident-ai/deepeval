import asyncio
from typing import Optional, Union, List

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.conversation_completeness.template import (
    ConversationCompletenessTemplate,
)
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
    convert_turn_to_dict,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import TurnParams
from deepeval.test_case.conversational_test_case import Turn
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.conversation_completeness.schema import *
from deepeval.metrics.api import metric_data_manager


class ConversationCompletenessMetric(BaseConversationalMetric):
    _required_test_case_params = [TurnParams.CONTENT, TurnParams.ROLE]

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
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
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
                self.user_intentions = self._extract_user_intentions(
                    test_case.turns
                )
                self.verdicts = [
                    self._generate_verdict(
                        turns=test_case.turns, intention=user_intention
                    )
                    for user_intention in self.user_intentions
                ]

                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Turns:\n{prettify_list(test_case.turns)}",
                        f"User Intentions:\n{prettify_list(self.user_intentions)}",
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
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            self.user_intentions = await self._a_extract_user_intentions(
                test_case.turns
            )
            self.verdicts = await asyncio.gather(
                *[
                    self._a_generate_verdict(
                        turns=test_case.turns, intention=user_intention
                    )
                    for user_intention in self.user_intentions
                ]
            )

            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Turns:\n{prettify_list(test_case.turns)}",
                    f"User Intentions:\n{prettify_list(self.user_intentions)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
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
            res, cost = await self.model.a_generate(
                prompt, schema=ConversationCompletenessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ConversationCompletenessScoreReason = (
                    await self.model.a_generate(
                        prompt, schema=ConversationCompletenessScoreReason
                    )
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

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
            res, cost = self.model.generate(
                prompt, schema=ConversationCompletenessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ConversationCompletenessScoreReason = self.model.generate(
                    prompt, schema=ConversationCompletenessScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdict(
        self, turns: List[Turn], intention: str
    ) -> ConversationCompletenessVerdict:
        prompt = ConversationCompletenessTemplate.generate_verdicts(
            turns=[convert_turn_to_dict(turn) for turn in turns],
            intention=intention,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ConversationCompletenessVerdict
            )
            self.evaluation_cost += cost
            return res
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
        self, turns: List[Turn], intention: str
    ) -> ConversationCompletenessVerdict:
        prompt = ConversationCompletenessTemplate.generate_verdicts(
            turns=[convert_turn_to_dict(turn) for turn in turns],
            intention=intention,
        )
        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=ConversationCompletenessVerdict
            )
            self.evaluation_cost += cost
            return res
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

    async def _a_extract_user_intentions(self, turns: List[Turn]) -> List[str]:
        prompt = ConversationCompletenessTemplate.extract_user_intentions(
            turns=[convert_turn_to_dict(turn) for turn in turns]
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=UserIntentions
            )
            self.evaluation_cost += cost
            return res.intentions
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

    def _extract_user_intentions(self, turns: List[Turn]) -> List[str]:
        prompt = ConversationCompletenessTemplate.extract_user_intentions(
            turns=[convert_turn_to_dict(turn) for turn in turns]
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=UserIntentions)
            self.evaluation_cost += cost
            return res.intentions
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
