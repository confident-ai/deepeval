import asyncio
from typing import Optional, Union, List, Type

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.base_metric import Verdict, YES_NO
from deepeval.metrics.utils import (
    generate_qag_verdict,
    a_generate_qag_verdict,
    score_qag_verdicts,
    check_conversational_test_case_params,
    construct_verbose_logs,
    initialize_model,
    initialize_system_one_model,
    convert_turn_to_dict,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
    SystemOneBinarySpec,
    SystemOneEvalSpec,
    parse_questions,
    run_system_one_eval,
    a_run_system_one_eval,
)
from deepeval.config.eval_mode import EvalModeName, resolve_eval_mode
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.test_case import ConversationalTestCase
from deepeval.test_case import MultiTurnParams
from deepeval.test_case.conversational_test_case import Turn
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.conversation_completeness.schema import (
    UserIntentions,
    ConversationCompletenessVerdict,
    ConversationCompletenessScoreReason,
)
from deepeval.templates import make_template_class


ConversationCompletenessTemplate = make_template_class(
    "ConversationCompletenessMetric"
)


class ConversationCompletenessMetric(BaseConversationalMetric):
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
        window_size: int = 3,
        flaky: bool = False,
        evaluation_template: Type[
            ConversationCompletenessTemplate
        ] = ConversationCompletenessTemplate,
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
        self.window_size = window_size
        self.evaluation_template = evaluation_template

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ):

        multimodal = test_case.multimodal
        check_conversational_test_case_params(
            test_case,
            self._required_test_case_params,
            self,
            False,
            self.model,
            multimodal,
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

                self.user_intentions = self._extract_user_intentions(
                    test_case.turns, multimodal=multimodal
                )
                self.verdicts = [
                    self._generate_verdict(
                        turns=test_case.turns,
                        intention=user_intention,
                        multimodal=multimodal,
                    )
                    for user_intention in self.user_intentions
                ]

                self.score = self._calculate_score()
                self.reason = self._generate_reason(multimodal=multimodal)
                self.success = self.is_successful()
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Turns:\n{prettify_list(test_case.turns)}",
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
        _in_component: bool = False,
    ) -> float:

        multimodal = test_case.multimodal
        check_conversational_test_case_params(
            test_case,
            self._required_test_case_params,
            self,
            False,
            self.model,
            multimodal,
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

            self.user_intentions = await self._a_extract_user_intentions(
                test_case.turns, multimodal=multimodal
            )
            self.verdicts = await asyncio.gather(
                *[
                    self._a_generate_verdict(
                        turns=test_case.turns,
                        intention=user_intention,
                        multimodal=multimodal,
                    )
                    for user_intention in self.user_intentions
                ]
            )

            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(multimodal=multimodal)
            self.success = self.is_successful()
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Turns:\n{prettify_list(test_case.turns)}",
                    f"User Intentions:\n{prettify_list(self.user_intentions)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        incompletenesses: List[str] = []
        for verdict in self.verdicts:
            if (
                verdict is not None
                and verdict.verdict is not None
                and verdict.verdict == Verdict.NO
            ):
                incompletenesses.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            score=self.score,
            incompletenesses=incompletenesses,
            intentions=self.user_intentions,
            multimodal=multimodal,
        )
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ConversationCompletenessScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    def _generate_reason(self, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        incompletenesses: List[str] = []
        for verdict in self.verdicts:
            if (
                verdict is not None
                and verdict.verdict is not None
                and verdict.verdict == Verdict.NO
            ):
                incompletenesses.append(verdict.reason)

        prompt = self._get_prompt(
            "generate_reason",
            score=self.score,
            incompletenesses=incompletenesses,
            intentions=self.user_intentions,
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ConversationCompletenessScoreReason,
            extract_schema=lambda score_reason: score_reason.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_verdict(
        self, turns: List[Turn], intention: str, multimodal: bool
    ) -> ConversationCompletenessVerdict:
        prompt = self._get_prompt(
            "generate_verdicts",
            turns=[convert_turn_to_dict(turn) for turn in turns],
            intention=intention,
            multimodal=multimodal,
        )

        return await a_generate_qag_verdict(
            metric=self,
            prompt=prompt,
            verdict_cls=ConversationCompletenessVerdict,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(
                turns, intention, multimodal
            ),
        )

    def _generate_verdict(
        self, turns: List[Turn], intention: str, multimodal: bool
    ) -> ConversationCompletenessVerdict:
        prompt = self._get_prompt(
            "generate_verdicts",
            turns=[convert_turn_to_dict(turn) for turn in turns],
            intention=intention,
            multimodal=multimodal,
        )
        return generate_qag_verdict(
            metric=self,
            prompt=prompt,
            verdict_cls=ConversationCompletenessVerdict,
            allowed=YES_NO,
            system_one=self._experimental_system_one_spec(
                turns, intention, multimodal
            ),
        )

    def _experimental_system_one_spec(
        self, turns: List[Turn], intention: str, multimodal: bool
    ) -> Optional[SystemOneBinarySpec]:
        if multimodal:
            return None
        return SystemOneBinarySpec(
            instructions=self._get_prompt("_experimental_system_one_verdict"),
            state={
                "turns": [convert_turn_to_dict(turn) for turn in turns],
                "intention": intention,
            },
        )

    def _system_one_eval_spec(
        self, test_case: ConversationalTestCase
    ) -> Optional[SystemOneEvalSpec]:
        """`system_one` eval mode: the whole conversation as one Jev
        request, each turn carrying its `role` and `content`; see
        EXPERIMENTAL.md."""
        if test_case.multimodal:
            return None
        return SystemOneEvalSpec(
            evaluation_params=self._required_test_case_params,
            questions=parse_questions(
                self._get_prompt("_experimental_system_one_questions")
            ),
        )

    async def _a_extract_user_intentions(
        self, turns: List[Turn], multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "extract_user_intentions",
            turns=[convert_turn_to_dict(turn) for turn in turns],
            multimodal=multimodal,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=UserIntentions,
            extract_schema=lambda r: r.intentions,
            extract_json=lambda data: UserIntentions(**data).intentions,
        )

    def _extract_user_intentions(
        self, turns: List[Turn], multimodal: bool
    ) -> List[str]:
        prompt = self._get_prompt(
            "extract_user_intentions",
            turns=[convert_turn_to_dict(turn) for turn in turns],
            multimodal=multimodal,
        )
        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=UserIntentions,
            extract_schema=lambda r: r.intentions,
            extract_json=lambda data: UserIntentions(**data).intentions,
        )

    def _calculate_score(self) -> float:
        # None verdicts (failed generation / out-of-vocabulary replies) are
        # dropped by score_qag_verdicts.
        return score_qag_verdicts(self, self.verdicts, passing=(Verdict.YES,))

    @property
    def __name__(self):
        return "Conversation Completeness"
