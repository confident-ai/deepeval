from typing import List, Optional, Union, Type, Tuple
import asyncio

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
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.turn_faithfulness.template import (
    TurnFaithfulnessTemplate,
)
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.turn_faithfulness.schema import (
    FaithfulnessVerdict,
    Verdicts,
    FaithfulnessScoreReason,
    Truths,
    Claims,
    InteractionFaithfulnessScore,
)
from deepeval.metrics.api import metric_data_manager


class TurnFaithfulnessMetric(BaseConversationalMetric):
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
        truths_extraction_limit: Optional[int] = None,
        penalize_ambiguous_claims: bool = False,
        evaluation_template: Type[
            TurnFaithfulnessTemplate
        ] = TurnFaithfulnessTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
        self.penalize_ambiguous_claims = penalize_ambiguous_claims

        self.truths_extraction_limit = truths_extraction_limit
        if self.truths_extraction_limit is not None:
            self.truths_extraction_limit = max(self.truths_extraction_limit, 0)

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
                scores = self._get_faithfulness_scores(
                    unit_interactions, multimodal
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
            scores = await self._a_get_faithfulness_scores(
                unit_interactions, multimodal
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

    async def _a_get_faithfulness_scores(
        self, unit_interactions: List[List[Turn]], multimodal: bool
    ):

        async def get_interaction_score(unit_interaction: List[Turn]):
            user_content = "User Message: "
            retrieval_context = []
            assistant_content = "Assistant Message: "
            for turn in unit_interaction:
                if turn.role == "user":
                    user_content += f"\n{turn.content} "
                else:
                    assistant_content += f"\n{turn.content} "
                    retrieval_context.extend(turn.retrieval_context)
            truths = await self._a_generate_truths(
                retrieval_context, multimodal
            )
            claims = await self._a_generate_claims(
                user_content, assistant_content, multimodal
            )
            verdicts = await self._a_generate_verdicts(
                claims, truths, multimodal
            )
            score, reason = self._get_interaction_score_and_reason(
                verdicts, multimodal
            )
            interaction_score = InteractionFaithfulnessScore(
                score=score,
                reason=reason,
                claims=claims,
                truths=truths,
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

    def _get_faithfulness_scores(
        self, unit_interactions: List[List[Turn]], multimodal: bool
    ):
        interaction_scores = []

        for unit_interaction in unit_interactions:
            user_content = "User Message: "
            retrieval_context = []
            assistant_content = "Assistant Message: "
            for turn in unit_interaction:
                if turn.role == "user":
                    user_content += f"\n{turn.content} "
                else:
                    assistant_content += f"\n{turn.content} "
                    retrieval_context.extend(turn.retrieval_context)
            truths = self._generate_truths(retrieval_context, multimodal)
            claims = self._generate_claims(
                user_content, assistant_content, multimodal
            )
            verdicts = self._generate_verdicts(claims, truths, multimodal)
            score, reason = self._get_interaction_score_and_reason(
                verdicts, multimodal
            )
            interaction_score = InteractionFaithfulnessScore(
                score=score,
                reason=reason,
                claims=claims,
                truths=truths,
                verdicts=verdicts,
            )
            interaction_scores.append(interaction_score)

        return interaction_scores

    async def _a_generate_truths(
        self, retrieval_context: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_truths(
            reference_context="\n\n".join(retrieval_context),
            extraction_limit=self.truths_extraction_limit,
            multimodal=multimodal,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Truths)
            self.evaluation_cost += cost
            return res.truths
        else:
            try:
                res: Truths = await self.model.a_generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    def _generate_truths(
        self, retrieval_context: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_truths(
            reference_context="\n\n".join(retrieval_context),
            extraction_limit=self.truths_extraction_limit,
            multimodal=multimodal,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Truths)
            self.evaluation_cost += cost
            return res.truths
        else:
            try:
                res: Truths = self.model.generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    async def _a_generate_claims(
        self, user_content: str, assistant_content: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_claims(
            input=user_content,
            assistant_output=assistant_content,
            multimodal=multimodal,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt, schema=Claims)
            self.evaluation_cost += cost
            return res.claims
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _generate_claims(
        self, user_content: str, assistant_content: str, multimodal: bool
    ) -> List[str]:
        prompt = self.evaluation_template.generate_claims(
            input=user_content,
            assistant_output=assistant_content,
            multimodal=multimodal,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=Claims)
            self.evaluation_cost += cost
            return res.claims
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    async def _a_generate_verdicts(
        self, claims: Claims, truths: Truths, multimodal: bool
    ) -> List[FaithfulnessVerdict]:
        if len(claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []

        prompt = self.evaluation_template.generate_verdicts(
            claims=claims,
            reference_context="\n\n".join(truths),
            multimodal=multimodal,
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
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(
        self, claims: Claims, truths: Truths, multimodal: bool
    ) -> List[FaithfulnessVerdict]:
        if len(claims) == 0:
            return []

        verdicts: List[FaithfulnessVerdict] = []

        prompt = self.evaluation_template.generate_verdicts(
            claims=claims,
            reference_context="\n\n".join(truths),
            multimodal=multimodal,
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
                    FaithfulnessVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _get_interaction_score_and_reason(
        self, verdicts, multimodal: bool
    ) -> Tuple[float, str]:
        number_of_verdicts = len(verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

            if (
                self.penalize_ambiguous_claims
                and verdict.verdict.strip().lower() == "idk"
            ):
                faithfulness_count -= 1

        score = faithfulness_count / number_of_verdicts
        reason = self._get_interaction_reason(score, verdicts, multimodal)
        return (
            (0, reason)
            if self.strict_mode and score < self.threshold
            else (score, reason)
        )

    async def _a_get_interaction_score_and_reason(
        self, verdicts, multimodal: bool
    ) -> Tuple[float, str]:
        number_of_verdicts = len(verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in verdicts:
            if verdict.verdict.strip().lower() != "no":
                faithfulness_count += 1

            if (
                self.penalize_ambiguous_claims
                and verdict.verdict.strip().lower() == "idk"
            ):
                faithfulness_count -= 1

        score = faithfulness_count / number_of_verdicts
        reason = await self._a_get_interaction_reason(
            score, verdicts, multimodal
        )
        return (
            (0, reason)
            if self.strict_mode and score < self.threshold
            else (score, reason)
        )

    async def _a_get_interaction_reason(
        self, score, verdicts, multimodal: bool
    ) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            contradictions=contradictions,
            score=format(score, ".2f"),
            multimodal=multimodal,
        )

        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=FaithfulnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: FaithfulnessScoreReason = await self.model.a_generate(
                    prompt, schema=FaithfulnessScoreReason
                )
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _get_interaction_reason(self, score, verdicts, multimodal: bool) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt = self.evaluation_template.generate_reason(
            contradictions=contradictions,
            score=format(score, ".2f"),
            multimodal=multimodal,
        )

        if self.using_native_model:
            res, cost = self.model.generate(
                prompt, schema=FaithfulnessScoreReason
            )
            self.evaluation_cost += cost
            return res.reason
        else:
            try:
                res: FaithfulnessScoreReason = self.model.generate(
                    prompt, schema=FaithfulnessScoreReason
                )
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _get_verbose_steps(
        self, interaction_scores: List[InteractionFaithfulnessScore]
    ):
        steps = []
        for index, interaction_score in enumerate(interaction_scores):
            interaction_steps = [
                f"Interaction {index + 1} \n",
                f"Truths: {prettify_list(interaction_score.truths)} \n",
                f"Claims: {prettify_list(interaction_score.claims)} \n",
                f"Verdicts: {prettify_list(interaction_score.verdicts)} \n",
                f"Score: {interaction_score.score} \n",
                f"Reason: {interaction_score.reason} \n",
            ]
            steps.extend(interaction_steps)
        return steps

    def _generate_reason(
        self, scores: List[InteractionFaithfulnessScore]
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
        self, scores: List[InteractionFaithfulnessScore]
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
        self, scores: List[InteractionFaithfulnessScore]
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
        return "Turn Faithfulness"
