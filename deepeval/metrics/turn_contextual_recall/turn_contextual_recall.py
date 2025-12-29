from typing import List, Optional, Union, Type, Tuple
import asyncio
import itertools
from deepeval.test_case import ConversationalTestCase, TurnParams, Turn
from deepeval.metrics import BaseConversationalMetric
from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_conversational_test_case_params,
    get_unit_interactions,
    get_turns_in_sliding_window,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.turn_contextual_recall.template import (
    TurnContextualRecallTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.turn_contextual_recall.schema import (
    ContextualRecallVerdict,
    Verdicts,
    ContextualRecallScoreReason,
    InteractionContextualRecallScore,
)
from deepeval.metrics.api import metric_data_manager


class TurnContextualRecallMetric(BaseConversationalMetric):
    _required_test_case_params: List[TurnParams] = [
        TurnParams.ROLE,
        TurnParams.CONTENT,
        TurnParams.RETRIEVAL_CONTEXT,
        TurnParams.EXPECTED_OUTCOME,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        window_size: int = 10,
        evaluation_template: Type[
            TurnContextualRecallTemplate
        ] = TurnContextualRecallTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.window_size = window_size
        self.evaluation_template = evaluation_template

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
            False,
            self.model,
            test_case.multimodal,
        )

        multimodal = test_case.multimodal

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
                unit_interactions = get_unit_interactions(test_case.turns)
                turns_windows: List[List[Turn]] = [
                    list(itertools.chain(*window))
                    for window in get_turns_in_sliding_window(
                        unit_interactions, self.window_size
                    )
                ]
                scores = []
                for window in turns_windows:
                    scores.extend(
                        self._get_contextual_recall_scores(
                            window, test_case.expected_outcome, multimodal
                        )
                    )
                self.score = self._calculate_score(scores)
                self.success = self.score >= self.threshold
                self.reason = self._generate_reason(scores)
                verbose_steps = self._get_verbose_steps(scores)
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        *verbose_steps,
                        f"Final Score: {self.score}\n",
                        f"Final Reason: {self.reason}\n",
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
            False,
            self.model,
            test_case.multimodal,
        )

        multimodal = test_case.multimodal

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            unit_interactions = get_unit_interactions(test_case.turns)
            turns_windows: List[List[Turn]] = [
                list(itertools.chain(*window))
                for window in get_turns_in_sliding_window(
                    unit_interactions, self.window_size
                )
            ]
            scores = []
            tasks = []

            async def get_individual_scores(window):
                scores.extend(
                    await self._a_get_contextual_recall_scores(
                        window, test_case.multimodal, multimodal
                    )
                )

            for window in turns_windows:
                tasks.append(get_individual_scores(window))
            await asyncio.gather(*tasks)
            self.score = self._calculate_score(scores)
            self.success = self.score >= self.threshold
            self.reason = await self._a_generate_reason(scores)
            verbose_steps = self._get_verbose_steps(scores)
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    *verbose_steps,
                    f"Final Score: {self.score}\n",
                    f"Final Reason: {self.reason}\n",
                ],
            )
            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

            return self.score

    async def _a_get_contextual_recall_scores(
        self,
        turns_window: List[Turn],
        expected_outcome: str,
        multimodal: bool,
    ):
        windows_scores = []

        user_content = ""
        retrieval_context = []
        for turn in turns_window:
            if turn.role == "user":
                user_content += f"\n{turn.content} "
            else:
                if turn.retrieval_context is not None:
                    retrieval_context.extend(turn.retrieval_context)

        verdicts = await self._a_generate_verdicts(
            expected_outcome, retrieval_context, multimodal
        )
        score, reason = await self._a_get_interaction_score_and_reason(
            expected_outcome, verdicts, multimodal
        )
        interaction_score = InteractionContextualRecallScore(
            score=score,
            reason=reason,
            verdicts=verdicts,
        )
        windows_scores.append(interaction_score)

        return windows_scores

    def _get_contextual_recall_scores(
        self,
        turns_window: List[Turn],
        expected_outcome: str,
        multimodal: bool,
    ):
        windows_scores = []

        user_content = ""
        retrieval_context = []
        for turn in turns_window:
            if turn.role == "user":
                user_content += f"\n{turn.content} "
            else:
                if turn.retrieval_context is not None:
                    retrieval_context.extend(turn.retrieval_context)

        verdicts = self._generate_verdicts(
            expected_outcome, retrieval_context, multimodal
        )
        score, reason = self._get_interaction_score_and_reason(
            expected_outcome, verdicts, multimodal
        )
        interaction_score = InteractionContextualRecallScore(
            score=score,
            reason=reason,
            verdicts=verdicts,
        )
        windows_scores.append(interaction_score)

        return windows_scores

    async def _a_generate_verdicts(
        self,
        expected_outcome: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[ContextualRecallVerdict]:
        if len(retrieval_context) == 0:
            return []

        verdicts: List[ContextualRecallVerdict] = []

        prompt = self.evaluation_template.generate_verdicts(
            expected_outcome=expected_outcome,
            retrieval_context=retrieval_context,
            multimodal=multimodal,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: s.verdicts,
            extract_json=lambda data: data["verdicts"],
        )

    def _generate_verdicts(
        self,
        expected_outcome: str,
        retrieval_context: List[str],
        multimodal: bool,
    ) -> List[ContextualRecallVerdict]:
        if len(retrieval_context) == 0:
            return []

        verdicts: List[ContextualRecallVerdict] = []

        prompt = self.evaluation_template.generate_verdicts(
            expected_outcome=expected_outcome,
            retrieval_context=retrieval_context,
            multimodal=multimodal,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=Verdicts,
            extract_schema=lambda s: s.verdicts,
            extract_json=lambda data: data["verdicts"],
        )

    async def _a_get_interaction_score_and_reason(
        self,
        expected_outcome: str,
        verdicts: List[ContextualRecallVerdict],
        multimodal: bool,
    ) -> Tuple[float, str]:
        if len(verdicts) == 0:
            return (
                1,
                "There were no retrieval contexts in the given turns to evaluate the contextual recall.",
            )

        score = self._calculate_interaction_score(verdicts)
        reason = await self._a_get_interaction_reason(
            expected_outcome, score, verdicts, multimodal
        )
        return (
            (0, reason)
            if self.strict_mode and score < self.threshold
            else (score, reason)
        )

    def _get_interaction_score_and_reason(
        self,
        expected_outcome: str,
        verdicts: List[ContextualRecallVerdict],
        multimodal: bool,
    ) -> Tuple[float, str]:
        if len(verdicts) == 0:
            return (
                1,
                "There were no retrieval contexts in the given turns to evaluate the contextual recall.",
            )

        score = self._calculate_interaction_score(verdicts)
        reason = self._get_interaction_reason(
            expected_outcome, score, verdicts, multimodal
        )
        return (
            (0, reason)
            if self.strict_mode and score < self.threshold
            else (score, reason)
        )

    def _calculate_interaction_score(
        self, verdicts: List[ContextualRecallVerdict]
    ) -> float:
        number_of_verdicts = len(verdicts)
        if number_of_verdicts == 0:
            return 1

        attributable_count = 0
        for verdict in verdicts:
            if verdict.verdict.strip().lower() == "yes":
                attributable_count += 1

        score = attributable_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_get_interaction_reason(
        self,
        expected_outcome: str,
        score: float,
        verdicts: List[ContextualRecallVerdict],
        multimodal: bool,
    ) -> str:
        if self.include_reason is False:
            return None

        # Prepare verdicts with node information for reasoning
        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            expected_outcome=expected_outcome,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(score, ".2f"),
            multimodal=multimodal,
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _get_interaction_reason(
        self,
        expected_outcome: str,
        score: float,
        verdicts: List[ContextualRecallVerdict],
        multimodal: bool,
    ) -> str:
        if self.include_reason is False:
            return None

        # Prepare verdicts with node information for reasoning
        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            expected_outcome=expected_outcome,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(score, ".2f"),
            multimodal=multimodal,
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _get_verbose_steps(
        self, interaction_scores: List[InteractionContextualRecallScore]
    ):
        steps = []
        for index, interaction_score in enumerate(interaction_scores):
            interaction_steps = [
                f"Window {index + 1} \n",
                f"Verdicts: {prettify_list(interaction_score.verdicts)} \n",
                f"Score: {interaction_score.score} \n",
                f"Reason: {interaction_score.reason} \n",
            ]
            steps.extend(interaction_steps)
        return steps

    def _generate_reason(
        self, scores: List[InteractionContextualRecallScore]
    ) -> str:
        if self.include_reason is False:
            return None

        if len(scores) == 0:
            return "There were no retrieval contexts in your turns to evaluate, hence the score is 1"

        reasons = []
        for score in scores:
            reasons.append(score.reason)

        prompt = self.evaluation_template.generate_final_reason(
            self.score, self.success, reasons
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    async def _a_generate_reason(
        self, scores: List[InteractionContextualRecallScore]
    ) -> str:
        if self.include_reason is False:
            return None

        if len(scores) == 0:
            return "There were no retrieval contexts in your turns to evaluate, hence the score is 1"

        reasons = []
        for score in scores:
            reasons.append(score.reason)

        prompt = self.evaluation_template.generate_final_reason(
            self.score, self.success, reasons
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ContextualRecallScoreReason,
            extract_schema=lambda s: s.reason,
            extract_json=lambda data: data["reason"],
        )

    def _calculate_score(
        self, scores: List[InteractionContextualRecallScore]
    ) -> float:
        number_of_scores = len(scores)
        if number_of_scores == 0:
            return 1
        total_score = 0
        for score in scores:
            total_score += score.score
        return total_score / number_of_scores

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Turn Contextual Recall"
