from typing import Optional, List
from pydantic import BaseModel, Field
import json

from deepeval.utils import trimToJson
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.models import GPTModel
from deepeval.templates import ContextualPrecisionTemplate


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    node: str = Field(default=None)


class ContextualPrecisionMetric(BaseMetric):
    def __init__(
        self,
        minimum_score: float = 0.5,
        model: Optional[str] = None,
    ):
        self.minimum_score = minimum_score
        self.model = model

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
        print(
            "✨ 🍰 ✨ You're using DeepEval's newest Contextual Precision Metric! This may take a minute."
        )
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

        self.success = contextual_precision_score >= self.minimum_score
        self.score = contextual_precision_score
        return self.score

    def _generate_reason(self, input: str, score: float):
        prompt = ContextualPrecisionTemplate.generate_reason(
            input=input,
            # Need to pass in entire verdict because the reason has to take into account
            # not just the relevant chunks, but the bad chunks.
            # for example, i can still have a perfect score with [1 1 0 0],
            # which then GPT will need the entire context to justify why the score is so high
            verdicts=self.verdicts,
            score=format(score, ".2f"),
        )
        chat_model = GPTModel(model_name=self.model)
        res = chat_model(prompt)

        return res.content

    def _generate_score(self):
        node_verdicts = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                node_verdicts.append(1)
            else:
                node_verdicts.append(0)

        sum_precision_at_k = 0.0
        relevant_nodes_count = 0

        # Go through each item in the response
        for k, is_relevant in enumerate(node_verdicts, start=1):
            # If the item is relevant, update the counter and add the precision at k to the sum
            if is_relevant:
                relevant_nodes_count += 1
                sum_precision_at_k += relevant_nodes_count / k

        # Calculate context precision
        context_precision = (
            sum_precision_at_k / relevant_nodes_count
            if relevant_nodes_count > 0
            else 0
        )
        return context_precision

    def _generate_verdicts(
        self, input: str, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        prompt = ContextualPrecisionTemplate.generate_verdicts(
            input=input,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        chat_model = GPTModel(model_name=self.model)
        res = chat_model(prompt)
        json_output = trimToJson(res.content)
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
        return self.success

    @property
    def __name__(self):
        return "Contextual Recall"


# response will be an array of in with only 0 and 1
def calculate_context_precision(response):
    """
    Calculate the context precision based on the provided response array.

    :param response: List[int], an array where 1 denotes relevant and 0 denotes not relevant.
    :return: float, the calculated context precision.
    """
    # Initialize the sum of precisions at each k
    sum_precision_at_k = 0.0
    # Counter for the number of relevant items found
    relevant_items_count = 0

    # Go through each item in the response
    for k, is_relevant in enumerate(response, start=1):
        # If the item is relevant, update the counter and add the precision at k to the sum
        if is_relevant:
            relevant_items_count += 1
            sum_precision_at_k += relevant_items_count / k

    # Calculate context precision
    context_precision = (
        sum_precision_at_k / relevant_items_count
        if relevant_items_count > 0
        else 0
    )
    return context_precision
