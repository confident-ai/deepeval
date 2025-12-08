from typing import List, Optional, Union, Type, Tuple
import asyncio

from deepeval.test_case import ConversationalTestCase, TurnParams, Turn
from deepeval.metrics import BaseMetric
from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_conversational_test_case_params,
    get_unit_interactions,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.turn_contextual_precision.template import (
    TurnContextualPrecisionTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.turn_contextual_precision.schema import (
    ContextualPrecisionVerdict,
    Verdicts,
    ContextualPrecisionScoreReason,
    InteractionContextualPrecisionScore,
)
from deepeval.metrics.api import metric_data_manager


class TurnContextualPrecisionMetric(BaseMetric):
    _required_test_case_params: List[TurnParams] = [
        TurnParams.CONTENT,
        TurnParams.RETRIEVAL_CONTEXT,
        
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[
            TurnContextualPrecisionTemplate
        ] = TurnContextualPrecisionTemplate,
        window_size: int = 10,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
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
        if test_case.expected_outcome is None:
            raise ValueError(
                "A test case must have the 'expected_outcome' populated to run the 'TurnContextualPrecisionMetric'"
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
                unit_interactions = get_unit_interactions(test_case.turns)
                scores = self._get_contextual_precision_scores(
                    unit_interactions, test_case.expected_outcome
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
            test_case, self._required_test_case_params, self
        )
        if test_case.expected_outcome is None:
            raise ValueError(
                "A test case must have the 'expected_outcome' populated to run the 'TurnContextualPrecisionMetric'"
            )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):
            unit_interactions = get_unit_interactions(test_case.turns)
            scores = await self._a_get_contextual_precision_scores(
                unit_interactions, test_case.expected_outcome
            )
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

    async def _a_get_contextual_precision_scores(
        self, unit_interactions: List[List[Turn]], _expected_outcome: str
    ):
        async def get_interaction_score(unit_interaction: List[Turn]):
            user_content = "User Message: "
            retrieval_context = []
            expected_outcome = f"Expected Assistant Message: \n{_expected_outcome}"
            for turn in unit_interaction:
                if turn.role == "user":
                    user_content += f"\n{turn.content} "
                else:
                    retrieval_context.extend(turn.retrieval_context)

            verdicts = await self._a_generate_verdicts(
                user_content, expected_outcome, retrieval_context
            )
            score, reason = await self._a_get_interaction_score_and_reason(
                user_content, verdicts
            )
            interaction_score = InteractionContextualPrecisionScore(
                score=score,
                reason=reason,
                verdicts=verdicts,
            )
            return interaction_score

        final_scores = await asyncio.gather(
            *[
                get_interaction_score(unit_interaction)
                for unit_interaction in unit_interactions
            ]
        )

        return final_scores

    def _get_contextual_precision_scores(
        self, unit_interactions: List[List[Turn]], _expected_outcome: str
    ):
        interaction_scores = []

        for unit_interaction in unit_interactions:
            user_content = "User Message: "
            retrieval_context = []
            expected_outcome = f"Expected Assistant Message: \n{_expected_outcome}"
            for turn in unit_interaction:
                if turn.role == "user":
                    user_content += f"\n{turn.content} "
                else:
                    retrieval_context.extend(turn.retrieval_context)

            verdicts = self._generate_verdicts(
                user_content, expected_outcome, retrieval_context
            )
            score, reason = self._get_interaction_score_and_reason(
                user_content, verdicts
            )
            interaction_score = InteractionContextualPrecisionScore(
                score=score,
                reason=reason,
                verdicts=verdicts,
            )
            interaction_scores.append(interaction_score)

        return interaction_scores

    async def _a_generate_verdicts(
        self, input: str, expected_outcome: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        if len(retrieval_context) == 0:
            return []

        verdicts: List[ContextualPrecisionVerdict] = []

        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            expected_outcome=expected_outcome,
            retrieval_context=retrieval_context,
            multimodal=False,
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    ContextualPrecisionVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(
        self, input: str, expected_outcome: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        if len(retrieval_context) == 0:
            return []

        verdicts: List[ContextualPrecisionVerdict] = []

        prompt = self.evaluation_template.generate_verdicts(
            input=input,
            expected_outcome=expected_outcome,
            retrieval_context=retrieval_context,
            multimodal=False,
        )

        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Verdicts)
            self.evaluation_cost += cost
            verdicts = [item for item in res.verdicts]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    ContextualPrecisionVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    async def _a_get_interaction_score_and_reason(
        self, input: str, verdicts: List[ContextualPrecisionVerdict]
    ) -> Tuple[float, str]:
        if len(verdicts) == 0:
            return 1, None

        score = self._calculate_interaction_score(verdicts)
        reason = await self._a_get_interaction_reason(input, score, verdicts)
        return (
            (0, reason)
            if self.strict_mode and score < self.threshold
            else (score, reason)
        )

    def _get_interaction_score_and_reason(
        self, input: str, verdicts: List[ContextualPrecisionVerdict]
    ) -> Tuple[float, str]:
        if len(verdicts) == 0:
            return 1, None

        score = self._calculate_interaction_score(verdicts)
        reason = self._get_interaction_reason(input, score, verdicts)
        return (
            (0, reason)
            if self.strict_mode and score < self.threshold
            else (score, reason)
        )

    def _calculate_interaction_score(
        self, verdicts: List[ContextualPrecisionVerdict]
    ) -> float:
        weighted_sum = 0
        total_weight = 0

        for i, verdict in enumerate(verdicts):
            # Rank starts at 1
            rank = i + 1
            # Calculate weight: 1/rank
            weight = 1 / rank

            if verdict.verdict.strip().lower() == "yes":
                weighted_sum += weight

            total_weight += weight

        if total_weight == 0:
            return 0

        # Weighted precision
        score = weighted_sum / total_weight
        return score

    async def _a_get_interaction_reason(
        self,
        input: str,
        score: float,
        verdicts: List[ContextualPrecisionVerdict],
    ) -> str:
        if self.include_reason is False:
            return None

        # Prepare verdicts with node information for reasoning
        verdicts_with_nodes = []
        for i, verdict in enumerate(verdicts):
            verdicts_with_nodes.append(
                {
                    "verdict": verdict.verdict,
                    "reason": verdict.reason,
                    "node": f"Node {i + 1}",
                }
            )

        prompt = self.evaluation_template.generate_reason(
            input=input,
            score=format(score, ".2f"),
            verdicts=verdicts_with_nodes,
            multimodal=False,
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=ContextualPrecisionScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ContextualPrecisionScoreReason = (
                    await self.model.a_generate(
                        prompt, schema=ContextualPrecisionScoreReason
                    )
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _get_interaction_reason(
        self,
        input: str,
        score: float,
        verdicts: List[ContextualPrecisionVerdict],
    ) -> str:
        if self.include_reason is False:
            return None

        # Prepare verdicts with node information for reasoning
        verdicts_with_nodes = []
        for i, verdict in enumerate(verdicts):
            verdicts_with_nodes.append(
                {
                    "verdict": verdict.verdict,
                    "reason": verdict.reason,
                    "node": f"Node {i + 1}",
                }
            )

        prompt = self.evaluation_template.generate_reason(
            input=input,
            score=format(score, ".2f"),
            verdicts=verdicts_with_nodes,
            multimodal=False,
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=ContextualPrecisionScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: ContextualPrecisionScoreReason = self.model.generate(
                    prompt, schema=ContextualPrecisionScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _get_verbose_steps(
        self, interaction_scores: List[InteractionContextualPrecisionScore]
    ):
        steps = []
        for index, interaction_score in enumerate(interaction_scores):
            interaction_steps = [
                f"Interaction {index + 1} \n",
                f"Verdicts: {prettify_list(interaction_score.verdicts)} \n",
                f"Score: {interaction_score.score} \n",
                f"Reason: {interaction_score.reason} \n",
            ]
            steps.extend(interaction_steps)
        return steps

    def _generate_reason(
        self, scores: List[InteractionContextualPrecisionScore]
    ) -> str:
        reasons = []
        for score in scores:
            reasons.append(score.reason)

        prompt = self.evaluation_template.generate_final_reason(
            self.score, self.success, reasons
        )

        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = self.model.generate(prompt)
            return res

    async def _a_generate_reason(
        self, scores: List[InteractionContextualPrecisionScore]
    ) -> str:
        reasons = []
        for score in scores:
            reasons.append(score.reason)

        prompt = self.evaluation_template.generate_final_reason(
            self.score, self.success, reasons
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = await self.model.a_generate(prompt)
            return res

    def _calculate_score(
        self, scores: List[InteractionContextualPrecisionScore]
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
        return "Turn Contextual Precision"