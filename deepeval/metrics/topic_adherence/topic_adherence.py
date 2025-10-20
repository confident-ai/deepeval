from typing import Optional, List, Tuple, Union, Dict

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    get_unit_interactions,
    check_conversational_test_case_params,
    initialize_model,
)
from deepeval.test_case import ConversationalTestCase, Turn, TurnParams
from deepeval.metrics.utils import print_tools_called
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator
from .template import TopicAdherenceTemplate
from .schema import RelevancyVerdict, QAPairs, QAPair


class TopicAdherenceMetric(BaseMetric):

    _required_test_case_params = [
        TurnParams.ROLE,
        TurnParams.CONTENT,
        TurnParams.TOOLS_CALLED,
    ]

    def __init__(
        self,
        relevant_topics: List[str],
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.relevant_topics = relevant_topics
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
                unit_interactions = get_unit_interactions(test_case.turns)
                interaction_pairs = self._get_qa_pairs(unit_interactions)
                True_Positives = [0, []]
                True_Negatives = [0, []]
                False_Positives = [0, []]
                False_Negatives = [0, []]
                for qa_pairs in interaction_pairs:
                    for qa_pair in qa_pairs.qa_pairs:
                        qa_verdict: RelevancyVerdict = self._get_qa_verdict(
                            qa_pair
                        )
                        if qa_verdict.verdict == "TP":
                            True_Positives[0] += 1
                            True_Positives[1].append(qa_verdict.reason)
                        elif qa_verdict.verdict == "TN":
                            True_Negatives[0] += 1
                            True_Negatives[1].append(qa_verdict.reason)
                        elif qa_verdict.verdict == "FP":
                            False_Positives[0] += 1
                            False_Positives[1].append(qa_verdict.reason)
                        elif qa_verdict.verdict == "FN":
                            False_Negatives[0] += 1
                            False_Negatives[1].append(qa_verdict.reason)
                self.normalize_scores(
                    True_Positives,
                    True_Negatives,
                    False_Positives,
                    False_Negatives,
                )
                
                # TODO: Use (TP + TN) / total instead of F1 score.
                # Reason is because, this allows us to evaluate an agent's handling of question instead of just focusing on F1

                precision, recall = self._get_precision_recall(
                    True_Positives,
                    True_Negatives,
                    False_Positives,
                    False_Negatives,
                )
                if precision + recall > 0:
                    self.score = (2 * precision * recall) / (precision + recall)
                else:
                    self.score = 0
                self.success = self.score >= self.threshold
                self.reason = self._generate_reason(
                    True_Positives,
                    True_Negatives,
                    False_Positives,
                    False_Negatives,
                )
                return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        check_conversational_test_case_params(
            test_case, self._required_test_case_params, self
        )

        self.evaluation_cost = 0 if self.using_native_model else None

        unit_interactions = get_unit_interactions(test_case.turns)
        interaction_pairs = await self._a_get_qa_pairs(unit_interactions)
        True_Positives = [0, []]
        True_Negatives = [0, []]
        False_Positives = [0, []]
        False_Negatives = [0, []]
        for qa_pairs in interaction_pairs:
            for qa_pair in qa_pairs.qa_pairs:
                qa_verdict: RelevancyVerdict = self._get_qa_verdict(qa_pair)
                if qa_verdict.verdict == "TP":
                    True_Positives[0] += 1
                    True_Positives[1].append(qa_verdict.reason)
                elif qa_verdict.verdict == "TN":
                    True_Negatives[0] += 1
                    True_Negatives[1].append(qa_verdict.reason)
                elif qa_verdict.verdict == "FP":
                    False_Positives[0] += 1
                    False_Positives[1].append(qa_verdict.reason)
                elif qa_verdict.verdict == "FN":
                    False_Negatives[0] += 1
                    False_Negatives[1].append(qa_verdict.reason)
        self.normalize_scores(
            True_Positives, True_Negatives, False_Positives, False_Negatives
        )
        precision, recall = self._get_precision_recall(
            True_Positives, True_Negatives, False_Positives, False_Negatives
        )
        if precision + recall > 0:
            self.score = (2 * precision * recall) / (precision + recall)
        else:
            self.score = 0
        self.success = self.score >= self.threshold
        self.reason = await self._a_generate_reason(
            True_Positives, True_Negatives, False_Positives, False_Negatives
        )
        return self.score

    def normalize_scores(
        self, True_Positives, True_Negatives, False_Positives, False_Negatives
    ):
        total = sum(
            [
                True_Positives[0],
                True_Negatives[0],
                False_Positives[0],
                False_Negatives[0],
            ]
        )
        True_Positives[0] /= total
        True_Negatives[0] /= total
        False_Positives[0] /= total
        False_Negatives[0] /= total
        True_Positives[0] += True_Negatives[0]

    def _generate_reason(self, TP, TN, FP, FN):
        prompt = TopicAdherenceTemplate.generate_reason(
            self.success, self.score, self.threshold, TP, TN, FP, FN
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = self.model.generate(prompt)
            return res

    async def _a_generate_reason(self, TP, TN, FP, FN):
        prompt = TopicAdherenceTemplate.generate_reason(
            self.success, self.score, self.threshold, TP, TN, FP, FN
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            return res
        else:
            res = await self.model.a_generate(prompt)
            return res

    def _get_precision_recall(self, TP, TN, FP, FN) -> Tuple[int, int]:
        precision_total = TP[0] + FP[0]
        recall_total = TP[0] + FN[0]
        precision = TP[0] / (precision_total if precision_total > 0 else 1)
        recall = TP[0] / (recall_total if recall_total > 0 else 1)
        return precision, recall

    def _get_qa_verdict(self, qa_pair: QAPair) -> RelevancyVerdict:
        prompt = TopicAdherenceTemplate.get_qa_pair_verdict(
            self.relevant_topics, qa_pair.question, qa_pair.response
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt, schema=RelevancyVerdict)
            self.evaluation_cost += cost
            return res
        else:
            try:
                res = self.model.generate(prompt, schema=RelevancyVerdict)
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return RelevancyVerdict(**data)

    async def _a_get_qa_verdict(self, qa_pair: QAPair) -> RelevancyVerdict:
        prompt = TopicAdherenceTemplate.get_qa_pair_verdict(
            self.relevant_topics, qa_pair.question, qa_pair.response
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt, schema=RelevancyVerdict
            )
            self.evaluation_cost += cost
            return res
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=RelevancyVerdict
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return RelevancyVerdict(**data)

    def _get_qa_pairs(self, unit_interactions: List) -> List[QAPairs]:
        qa_pairs = []
        for unit_interaction in unit_interactions:
            conversation = "Conversation: \n"
            for turn in unit_interaction:
                conversation += f"{turn.role} \n"
                conversation += f"{turn.content} \n\n"
            prompt = TopicAdherenceTemplate.get_qa_pairs(conversation)
            new_pair = None

            if self.using_native_model:
                res, cost = self.model.generate(prompt, schema=QAPairs)
                self.evaluation_cost += cost
                new_pair = res
            else:
                try:
                    res = self.model.generate(prompt, schema=QAPairs)
                    new_pair = res
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    new_pair = QAPairs(**data)

            if new_pair is not None:
                qa_pairs.append(new_pair)

        return qa_pairs

    async def _a_get_qa_pairs(self, unit_interactions: List) -> List[QAPairs]:
        qa_pairs = []
        for unit_interaction in unit_interactions:
            conversation = "Conversation: \n"
            for turn in unit_interaction:
                conversation += f"{turn.role} \n"
                conversation += f"{turn.content} \n\n"
            prompt = TopicAdherenceTemplate.get_qa_pairs(conversation)
            new_pair = None

            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt, schema=QAPairs)
                self.evaluation_cost += cost
                new_pair = res
            else:
                try:
                    res = await self.model.a_generate(prompt, schema=QAPairs)
                    new_pair = res
                except TypeError:
                    res = await self.model.a_generate(prompt)
                    data = trimAndLoadJson(res, self)
                    new_pair = QAPairs(**data)

            if new_pair is not None:
                qa_pairs.append(new_pair)

        return qa_pairs

    @property
    def __name__(self):
        return "Topic Adherence"
