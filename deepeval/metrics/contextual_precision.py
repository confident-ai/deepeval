from typing import Optional, List, Union
from pydantic import BaseModel, Field
import json
from langchain_core.language_models import BaseChatModel

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel, DeepEvalBaseModel
from deepeval.metrics.templates import ContextualPrecisionTemplate
from deepeval.progress_context import metrics_progress_context


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: str
    node: str = Field(default=None)


class ContextualPrecisionMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseModel, BaseChatModel]] = None,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        if isinstance(model, DeepEvalBaseModel):
            self.model = model
        else:
            self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase) -> float:
        if (
            test_case.input is None
            or test_case.actual_output is None
            or test_case.retrieval_context is None
            or test_case.expected_output is None
        ):
            raise ValueError(
                "Input, actual output, expected output, or retrieval context cannot be None"
            )

        with metrics_progress_context(self.__name__, self.evaluation_model):
            self.verdicts: List[
                ContextualPrecisionVerdict
            ] = self._generate_verdicts(
                test_case.input,
                test_case.expected_output,
                test_case.retrieval_context,
            )
            contextual_precision_score = self._generate_score()

            self.reason = self._generate_reason(
                test_case.input, contextual_precision_score
            )

            self.success = contextual_precision_score >= self.threshold
            self.score = contextual_precision_score
            return self.score

    def _generate_reason(self, input: str, score: float):
        if self.include_reason is False:
            return None

        retrieval_contexts_verdicts = [
            {
                "verdict": verdict.verdict,
                "reasons": verdict.reason,
                "node": verdict.node,
            }
            for verdict in self.verdicts
        ]

        prompt = ContextualPrecisionTemplate.generate_reason(
            input=input,
            # Need to pass in entire verdict because the reason has to take into account
            # not just the relevant chunks, but the bad chunks.
            # for example, i can still have a perfect score with [1 1 0 0],
            # which then GPT will need the entire context to justify why the score is so high
            verdicts=retrieval_contexts_verdicts,
            score=format(score, ".2f"),
        )

        res = self.model(prompt)
        return res

    def _generate_score(self):
        # Convert verdicts to a binary list where 'yes' is 1 and others are 0
        node_verdicts = [
            1 if v.verdict.strip().lower() == "yes" else 0
            for v in self.verdicts
        ]

        sum_weighted_precision_at_k = 0.0
        relevant_nodes_count = 0

        # Go through each item in the response
        for k, is_relevant in enumerate(node_verdicts, start=1):
            # If the item is relevant, update the counter and add the weighted precision at k to the sum
            if is_relevant:
                relevant_nodes_count += 1
                precision_at_k = relevant_nodes_count / k
                sum_weighted_precision_at_k += precision_at_k * is_relevant

        # Calculate weighted cumulative precision
        if relevant_nodes_count == 0:
            return 0

        weighted_cumulative_precision = (
            sum_weighted_precision_at_k / relevant_nodes_count
        )

        return weighted_cumulative_precision

    def _generate_verdicts(
        self, input: str, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        prompt = ContextualPrecisionTemplate.generate_verdicts(
            input=input,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        res = self.model(prompt)
        json_output = trimToJson(res)
        data = json.loads(json_output)
        verdicts = [
            ContextualPrecisionVerdict(**item) for item in data["verdicts"]
        ]

        if len(verdicts) != len(retrieval_context):
            raise ValueError(
                "Number of verdicts generated does not equal length of retrieved nodes."
            )

        for i in range(len(verdicts)):
            verdicts[i].node = retrieval_context[i]

        return verdicts

    def is_successful(self) -> bool:
        self.success = self.score >= self.threshold
        return self.success

    @property
    def __name__(self):
        return "Contextual Precision"
