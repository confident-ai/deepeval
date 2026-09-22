"""JevEval: a score metric whose decision points are answered by TypeSafe AI's
System One model (Jev) with calibrated probabilities, never generated text.

The user defines the decision points as ``Noul`` / ``Score`` / ``Choice``
questions. DeepEval sends them to Jev in one ``decide()`` call, maps each
answer onto ``[0, 1]`` and takes a weighted mean. The evaluation LLM is used
for exactly one optional thing: writing a reason grounded in the test case.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Union

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    a_generate_with_schema_and_extract,
    accrue_token_usage,
    check_llm_test_case_params,
    construct_verbose_logs,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.utils import get_or_create_event_loop
from deepeval.config.settings import get_settings

from . import schema as jschema
from .questions import JevQuestion, QuestionOutcome
from .utils import (
    aggregate,
    build_questions,
    construct_single_turn_state,
    describe_outcomes,
    format_outcomes_for_logs,
    initialize_jev_model,
    initialize_reason_model,
    min_confidence,
    outcomes_from_answers,
    validate_questions,
)


class JevEval(BaseMetric):
    def __init__(
        self,
        name: str,
        evaluation_params: Optional[List[SingleTurnParams]] = None,
        questions: Optional[Sequence[JevQuestion]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        threshold: Optional[float] = 0.5,
        async_mode: bool = True,
        verbose_mode: bool = False,
        flaky: bool = False,
        _include_jev_eval_suffix: bool = True,
    ):
        if not evaluation_params:
            raise ValueError(
                "evaluation_params cannot be empty; list the test case fields "
                "your questions refer to."
            )
        self.name = name
        self.evaluation_params = list(evaluation_params)
        self.questions = validate_questions(questions)
        self.system_one_model = initialize_jev_model(system_one_model)
        self.include_reason = include_reason
        self.model, self.using_native_model = initialize_reason_model(
            model, include_reason
        )
        self.evaluation_model = self.system_one_model.get_model_name()
        if include_reason:
            self.evaluation_model += f" + {self.model.get_model_name()}"
        self.threshold = threshold
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.flaky = flaky
        self._include_jev_eval_suffix = _include_jev_eval_suffix
        self.score_breakdown: Optional[List[Dict[str, Any]]] = None
        self.confidence: Optional[float] = None

    ###############################################
    # Measure
    ###############################################

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self.evaluation_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )
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
                state = construct_single_turn_state(
                    self.evaluation_params, test_case
                )
                outcomes = self._decide(state)
                self.reason = (
                    self._generate_reason(state, outcomes, multimodal)
                    if self.include_reason
                    else None
                )
                self._finalize(outcomes)
            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self.evaluation_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )
        self._reset_cost()

        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            state = construct_single_turn_state(
                self.evaluation_params, test_case
            )
            outcomes = await self._a_decide(state)
            self.reason = (
                await self._a_generate_reason(state, outcomes, multimodal)
                if self.include_reason
                else None
            )
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
    # Reason (LLM, optional)
    ###############################################

    def _reason_prompt(
        self,
        state: Dict[str, Any],
        outcomes: List[QuestionOutcome],
        multimodal: bool,
    ) -> str:
        return self._get_prompt(
            "generate_reason",
            outcomes=describe_outcomes(self.questions, outcomes),
            test_case_content=json.dumps(
                state.get("test_case", {}), indent=2, ensure_ascii=False
            ),
            parameters=", ".join(p.value for p in self.evaluation_params),
            multimodal=multimodal,
        )

    def _generate_reason(
        self,
        state: Dict[str, Any],
        outcomes: List[QuestionOutcome],
        multimodal: bool,
    ) -> str:
        return generate_with_schema_and_extract(
            metric=self,
            prompt=self._reason_prompt(state, outcomes, multimodal),
            schema_cls=jschema.Reason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda d: d["reason"],
        )

    async def _a_generate_reason(
        self,
        state: Dict[str, Any],
        outcomes: List[QuestionOutcome],
        multimodal: bool,
    ) -> str:
        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=self._reason_prompt(state, outcomes, multimodal),
            schema_cls=jschema.Reason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda d: d["reason"],
        )

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
        self.score = aggregate(outcomes)
        self.score_breakdown = [o.model_dump() for o in outcomes]
        self.confidence = min_confidence(outcomes)
        self.success = self.is_successful()
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Questions:\n{format_outcomes_for_logs(outcomes)}",
                f"Score: {self.score}",
                f"Reason: {self.reason}",
            ],
        )

    @property
    def __name__(self):
        if self._include_jev_eval_suffix:
            return f"{self.name} [JevEval]"
        return self.name
