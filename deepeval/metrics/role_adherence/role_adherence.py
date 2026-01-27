from typing import Optional, Union, List

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.api import metric_data_manager
from deepeval.metrics.role_adherence.schema import (
    OutOfCharacterResponseVerdicts,
    RoleAdherenceScoreReason,
)
from deepeval.metrics.role_adherence.template import RoleAdherenceTemplate
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    convert_turn_to_dict,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import Turn, ConversationalTestCase, TurnParams
from deepeval.utils import get_or_create_event_loop, prettify_list


class RoleAdherenceMetric(BaseConversationalMetric):
    _required_test_case_params = [TurnParams.CONTENT, TurnParams.ROLE]

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
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ):
        check_conversational_test_case_params(
            test_case,
            self._required_test_case_params,
            self,
            True,
            self.model,
            test_case.multimodal,
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
                self.out_of_character_verdicts: (
                    OutOfCharacterResponseVerdicts
                ) = self._extract_out_of_character_verdicts(
                    test_case.turns, test_case.chatbot_role
                )
                self.score = self._calculate_score(test_case.turns)
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
            test_case,
            self._required_test_case_params,
            self,
            True,
            self.model,
            test_case.multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            self.out_of_character_verdicts = (
                await (
                    self._a_extract_out_of_character_verdicts(
                        test_case.turns, test_case.chatbot_role
                    )
                )
            )
            self.score = self._calculate_score(test_case.turns)
            self.reason = await self._a_generate_reason(
                role=test_case.chatbot_role
            )
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Chatbot Role:\n{test_case.chatbot_role}",
                    f"Out-of-Character Turn(s):\n{prettify_list(self.out_of_character_verdicts.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )
            return self.score

    async def _a_generate_reason(self, role: str) -> Optional[str]:
        if self.include_reason is False:
            return None

        prompt = RoleAdherenceTemplate.generate_reason(
            score=self.score,
            role=role,
            out_of_character_responses=[
                verdict.ai_message
                for verdict in self.out_of_character_verdicts.verdicts
            ],
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=RoleAdherenceScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, role: str) -> Optional[str]:
        if self.include_reason is False:
            return None
        prompt = RoleAdherenceTemplate.generate_reason(
            score=self.score,
            role=role,
            out_of_character_responses=[
                verdict.ai_message
                for verdict in self.out_of_character_verdicts.verdicts
            ],
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=RoleAdherenceScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_extract_out_of_character_verdicts(
        self, turns: List[Turn], role: str
    ) -> OutOfCharacterResponseVerdicts:
        prompt = (
            RoleAdherenceTemplate.extract_out_of_character_response_verdicts(
                turns=[convert_turn_to_dict(turn) for turn in turns],
                role=role,
            )
        )
        res: OutOfCharacterResponseVerdicts = (
            await a_generate_with_schema_and_extract(
                metric=self,
                prompt=prompt,
                schema_cls=OutOfCharacterResponseVerdicts,
                extract_schema=lambda s: s,
                extract_json=lambda data: OutOfCharacterResponseVerdicts(
                    **data
                ),
            )
        )

        for verdict in res.verdicts:
            try:
                index = verdict.index
                verdict.ai_message = f"{turns[index].content} (turn #{index+1})"
            except Exception:
                pass
        return res

    def _extract_out_of_character_verdicts(
        self, turns: List[Turn], role: str
    ) -> OutOfCharacterResponseVerdicts:
        prompt = (
            RoleAdherenceTemplate.extract_out_of_character_response_verdicts(
                turns=[convert_turn_to_dict(turn) for turn in turns],
                role=role,
            )
        )
        res: OutOfCharacterResponseVerdicts = generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=OutOfCharacterResponseVerdicts,
            extract_schema=lambda s: s,
            extract_json=lambda data: OutOfCharacterResponseVerdicts(**data),
        )

        for verdict in res.verdicts:
            try:
                index = verdict.index
                verdict.ai_message = f"{turns[index].content} (turn #{index+1})"
            except Exception:
                pass
        return res

    def _calculate_score(self, turns: List[Turn]) -> float:
        number_of_turns = 0
        for turn in turns:
            if turn.role == "assistant":
                number_of_turns += 1
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
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Role Adherence"
