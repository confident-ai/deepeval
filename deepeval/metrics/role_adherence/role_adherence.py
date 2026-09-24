import asyncio
from typing import Dict, Optional, Union, List, Type

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.role_adherence.schema import (
    OutOfCharacterResponseVerdict,
    OutOfCharacterResponseVerdicts,
    RoleAdherenceScoreReason,
)
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    construct_verbose_logs,
    convert_turn_to_dict,
    initialize_model,
    initialize_system_one_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneBinarySpec,
    SystemOneEvalSpec,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
    system_one_probability,
    a_system_one_probability,
    verdict_from_probability,
)
from deepeval.metrics.base_metric import Verdict
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.templates import make_template_class


RoleAdherenceTemplate = make_template_class("RoleAdherenceMetric")


class RoleAdherenceMetric(BaseConversationalMetric):
    _required_test_case_params = [MultiTurnParams.CONTENT, MultiTurnParams.ROLE]

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[EvalModeName] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[
            RoleAdherenceTemplate
        ] = RoleAdherenceTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.eval_mode = resolve_eval_mode(eval_mode)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            system_one_model, self.eval_mode
        )
        self.evaluation_model = (
            self.model or self.system_one_model
        ).get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
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
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
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
                    )
                )
            else:
                if run_system_one_eval(self, test_case):
                    return self.score

                self.out_of_character_verdicts: (
                    OutOfCharacterResponseVerdicts
                ) = self._extract_out_of_character_verdicts(
                    test_case.turns,
                    test_case.chatbot_role,
                    test_case.multimodal,
                )
                self.score = self._calculate_score(test_case.turns)
                self.reason = self._generate_reason(role=test_case.chatbot_role)
                self.success = self.is_successful()
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
        _in_component: bool = False,
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
        self.input_tokens = 0 if self.using_native_model else None
        self.output_tokens = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            if await a_run_system_one_eval(self, test_case):
                return self.score

            self.out_of_character_verdicts = (
                await (
                    self._a_extract_out_of_character_verdicts(
                        test_case.turns,
                        test_case.chatbot_role,
                        test_case.multimodal,
                    )
                )
            )
            self.score = self._calculate_score(test_case.turns)
            self.reason = await self._a_generate_reason(
                role=test_case.chatbot_role
            )
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Chatbot Role:\n{test_case.chatbot_role}",
                    f"Out-of-Character Turn(s):\n{prettify_list(self.out_of_character_verdicts.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, role: str) -> Optional[str]:
        if self.include_reason is False:
            return None

        prompt = self._get_prompt(
            "generate_reason",
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
        prompt = self._get_prompt(
            "generate_reason",
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
        self, turns: List[Turn], role: str, multimodal: bool
    ) -> OutOfCharacterResponseVerdicts:
        specs = self._experimental_system_one_specs(turns, role, multimodal)
        if specs:
            probabilities = await asyncio.gather(
                *[
                    a_system_one_probability(self, spec)
                    for spec in specs.values()
                ]
            )
            if all(p is not None for p in probabilities):
                return self._system_one_verdicts(
                    turns, dict(zip(specs.keys(), probabilities))
                )

        prompt = self._get_prompt(
            "extract_out_of_character_response_verdicts",
            turns=[convert_turn_to_dict(turn) for turn in turns],
            role=role,
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
        self, turns: List[Turn], role: str, multimodal: bool
    ) -> OutOfCharacterResponseVerdicts:
        specs = self._experimental_system_one_specs(turns, role, multimodal)
        if specs:
            probabilities = {}
            for index, spec in specs.items():
                p = system_one_probability(self, spec)
                if p is None:
                    break
                probabilities[index] = p
            else:
                return self._system_one_verdicts(turns, probabilities)

        prompt = self._get_prompt(
            "extract_out_of_character_response_verdicts",
            turns=[convert_turn_to_dict(turn) for turn in turns],
            role=role,
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

    def _experimental_system_one_specs(
        self, turns: List[Turn], role: str, multimodal: bool
    ) -> Optional[Dict[int, SystemOneBinarySpec]]:
        if multimodal:
            return None
        instructions = self._get_prompt("_experimental_system_one_verdict")
        return {
            index: SystemOneBinarySpec(
                instructions=instructions,
                state={
                    "chatbot_role": role,
                    "previous_turns": [
                        convert_turn_to_dict(turn) for turn in turns[:index]
                    ],
                    "ai_message": turn.content,
                },
            )
            for index, turn in enumerate(turns)
            if turn.role == "assistant"
        }

    def _system_one_verdicts(
        self, turns: List[Turn], probabilities: Dict[int, float]
    ) -> OutOfCharacterResponseVerdicts:
        return OutOfCharacterResponseVerdicts(
            verdicts=[
                OutOfCharacterResponseVerdict(
                    index=index,
                    reason=f"P(in character)={p:.2f}",
                    ai_message=f"{turns[index].content} (turn #{index+1})",
                )
                for index, p in probabilities.items()
                if verdict_from_probability(p) == Verdict.NO
            ]
        )

    def _system_one_eval_spec(
        self, test_case: ConversationalTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole conversation and the
        `chatbot_role` as one Jev request, each turn carrying its `role` and
        `content`; see EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=[
                *self._required_test_case_params,
                MultiTurnParams.CHATBOT_ROLE,
            ],
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

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

    @property
    def __name__(self):
        return "Role Adherence"
