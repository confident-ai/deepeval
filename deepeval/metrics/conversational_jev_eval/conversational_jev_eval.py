"""Multi-turn counterpart of ``JevEval``: the same Jev-answered questions over
a ``ConversationalTestCase``, whose turns become the ``turns`` key of the
state Jev decides about. No LLM is involved; the reason is deterministic
text built from Jev's answers."""

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    accrue_token_usage,
    check_conversational_test_case_params,
    construct_verbose_logs,
    format_system_one_reason,
)
from deepeval.metrics.jev_eval.questions import JevQuestion, QuestionOutcome
from deepeval.metrics.jev_eval.utils import (
    aggregate,
    aggregate_strict,
    build_questions,
    construct_multi_turn_state,
    format_outcomes_for_logs,
    initialize_jev_model,
    mark_strict,
    min_confidence,
    outcomes_from_answers,
    validate_questions,
)
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.test_case import ConversationalTestCase, MultiTurnParams
from deepeval.utils import get_or_create_event_loop
from deepeval.config.settings import get_settings


class ConversationalJevEval(BaseConversationalMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: Optional[List[MultiTurnParams]] = None,
        questions: Optional[Sequence[JevQuestion]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        include_reason: bool = True,
        threshold: Optional[float] = 0.5,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose_mode: bool = False,
        flaky: bool = False,
        _include_jev_eval_suffix: bool = True,
    ):
        if not evaluation_params:
            raise ValueError(
                "evaluation_params cannot be empty; list the turn and "
                "conversation fields your questions refer to."
            )
        params = list(evaluation_params)
        # Every question is about the conversation, so the turns' content and
        # roles are always part of the state.
        if MultiTurnParams.CONTENT not in params:
            params.append(MultiTurnParams.CONTENT)
        if MultiTurnParams.ROLE not in params:
            params.append(MultiTurnParams.ROLE)

        self.name = name
        self.evaluation_params = params
        self.questions = validate_questions(questions)
        self.system_one_model = initialize_jev_model(system_one_model)
        self.include_reason = include_reason
        self.evaluation_model = self.system_one_model.get_model_name()
        self.strict_mode = strict_mode
        self.threshold = 1 if strict_mode else threshold
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self._include_jev_eval_suffix = _include_jev_eval_suffix
        self.score_breakdown: Optional[List[Dict[str, Any]]] = None
        self.confidence: Optional[float] = None

    ###############################################
    # Measure
    ###############################################

    def _check_params(self, test_case: ConversationalTestCase) -> None:
        if test_case.multimodal:
            raise ValueError(
                f"{self.__name__} evaluates text only: Jev has no image "
                "input. Pass a text-only ConversationalTestCase."
            )
        check_conversational_test_case_params(
            test_case,
            self.evaluation_params,
            self,
            MultiTurnParams.CHATBOT_ROLE in self.evaluation_params,
            None,
            False,
        )

    def measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self._check_params(test_case)
        self._reset_cost()

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                settings = get_settings()
                loop.run_until_complete(
                    asyncio.wait_for(
                        self.a_measure(
                            test_case,
                            _show_indicator=False,
                            _in_component=_in_component,
                        ),
                        timeout=(
                            None
                            if settings.DEEPEVAL_DISABLE_TIMEOUTS
                            else settings.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS
                        ),
                    )
                )
            else:
                state = construct_multi_turn_state(
                    self.evaluation_params, test_case
                )
                outcomes = self._decide(state)
                self._finalize(outcomes)
            return self.score

    async def a_measure(
        self,
        test_case: ConversationalTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        self._check_params(test_case)
        self._reset_cost()

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            state = construct_multi_turn_state(
                self.evaluation_params, test_case
            )
            outcomes = await self._a_decide(state)
            self._finalize(outcomes)
            return self.score

    ###############################################
    # Decision (Jev)
    ###############################################

    def _decide(self, state: Dict[str, Any]) -> List[QuestionOutcome]:
        answers, cost = self.system_one_model.decide(
            state, build_questions(self.questions)
        )
        self._accrue(cost)
        return outcomes_from_answers(self.questions, answers)

    async def _a_decide(self, state: Dict[str, Any]) -> List[QuestionOutcome]:
        answers, cost = await self.system_one_model.a_decide(
            state, build_questions(self.questions)
        )
        self._accrue(cost)
        return outcomes_from_answers(self.questions, answers)

    ###############################################
    # Reason (deterministic, no model call)
    ###############################################

    def _generate_reason(self, outcomes: List[QuestionOutcome]) -> str:
        if self.include_reason is False:
            return None
        return format_system_one_reason(self, outcomes)

    ###############################################
    # Bookkeeping
    ###############################################

    def _reset_cost(self) -> None:
        self.evaluation_cost = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _accrue(self, cost: Any) -> None:
        self._accrue_cost(cost)
        accrue_token_usage(self, cost)

    def _finalize(self, outcomes: List[QuestionOutcome]) -> None:
        if self.strict_mode:
            outcomes = mark_strict(self.questions, outcomes)
            self.score = aggregate_strict(outcomes)
        else:
            self.score = aggregate(outcomes)
        self._system_one_outcomes = outcomes
        self.score_breakdown = [o.model_dump() for o in outcomes]
        self.confidence = min_confidence(outcomes)
        self.reason = self._generate_reason(outcomes)
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Questions:\n{format_outcomes_for_logs(outcomes)}",
                f"Score: {self.score}\nConfidence: {self.confidence}",
                f"Reason: {self.reason}",
            ],
        )

    @property
    def __name__(self):
        if self._include_jev_eval_suffix:
            return f"{self.name} [Conversational JevEval]"
        return self.name
