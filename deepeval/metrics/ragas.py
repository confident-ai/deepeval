"""An implementation of the Ragas metric
"""
from typing import Optional, Union
from ragas.llms import LangchainLLM
from langchain_core.language_models import BaseChatModel

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel


def format_ragas_metric_name(name: str):
    return f"{name} (ragas)"


class RAGASContextualPrecisionMetric(BaseMetric):
    """This metric checks the contextual precision using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = None,
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.metrics import context_precision

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        context_precision.llm = LangchainLLM(llm=chat_model)

        # Create a dataset from the test case
        data = {
            "contexts": [test_case.retrieval_context],
            "question": [test_case.input],
            "ground_truths": [[test_case.expected_output]],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)

        # Evaluate the dataset using Ragas
        scores = evaluate(dataset, metrics=[context_precision])

        # Ragas only does dataset-level comparisons
        context_precision_score = scores["context_precision"]
        self.success = context_precision_score >= self.threshold
        self.score = context_precision_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Precision")


class RAGASContextualRelevancyMetric(BaseMetric):
    """This metric checks the contextual relevancy using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.metrics import context_relevancy

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        context_relevancy.llm = LangchainLLM(llm=chat_model)

        # Create a dataset from the test case
        data = {
            "contexts": [test_case.retrieval_context],
            "question": [test_case.input],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)

        # Evaluate the dataset using Ragas
        scores = evaluate(dataset, metrics=[context_relevancy])

        # Ragas only does dataset-level comparisons
        context_relevancy_score = scores["context_relevancy"]
        self.success = context_relevancy_score >= self.threshold
        self.score = context_relevancy_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Relevancy")


class RAGASAnswerRelevancyMetric(BaseMetric):
    """This metric checks the answer relevancy using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevancy

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        answer_relevancy.llm = LangchainLLM(llm=chat_model)

        data = {
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "contexts": [test_case.retrieval_context],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, metrics=[answer_relevancy])
        answer_relevancy_score = scores["answer_relevancy"]
        self.success = answer_relevancy_score >= self.threshold
        self.score = answer_relevancy_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Answer Relevancy")


class RAGASFaithfulnessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        faithfulness.llm = LangchainLLM(llm=chat_model)

        data = {
            "contexts": [test_case.retrieval_context],
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, metrics=[faithfulness])
        faithfulness_score = scores["faithfulness"]
        self.success = faithfulness_score >= self.threshold
        self.score = faithfulness_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Faithfulness")


class RAGASContextualRecallMetric(BaseMetric):
    """This metric checks the context recall using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.metrics import context_recall

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        context_recall.llm = LangchainLLM(llm=chat_model)

        data = {
            "question": [test_case.input],
            "ground_truths": [[test_case.expected_output]],
            "contexts": [test_case.retrieval_context],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, [context_recall])
        context_recall_score = scores["context_recall"]
        self.success = context_recall_score >= self.threshold
        self.score = context_recall_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Recall")


class HarmfulnessMetric(BaseMetric):
    """This metric checks the harmfulness using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
            from ragas.metrics.critique import harmfulness

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        harmfulness.llm = LangchainLLM(llm=chat_model)

        data = {
            "ground_truths": [[test_case.expected_output]],
            "contexts": [test_case.context],
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, [harmfulness])
        harmfulness_score = scores["harmfulness"]
        self.success = harmfulness_score >= self.threshold
        self.score = harmfulness_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Harmfulness"


class CoherenceMetric(BaseMetric):
    """This metric checks the coherence using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        try:
            from ragas import evaluate
            from ragas.metrics.critique import coherence
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        coherence.llm = LangchainLLM(llm=chat_model)

        data = {
            "ground_truths": [[test_case.expected_output]],
            "contexts": [test_case.context],
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, [coherence])
        coherence_score = scores["coherence"]
        self.success = coherence_score >= self.threshold
        self.score = coherence_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Coherence"


class MaliciousnessMetric(BaseMetric):
    """This metric checks the maliciousness using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        try:
            from ragas import evaluate
            from ragas.metrics.critique import maliciousness

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        maliciousness.llm = LangchainLLM(llm=chat_model)

        data = {
            "ground_truths": [[test_case.expected_output]],
            "contexts": [test_case.context],
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, [maliciousness])
        maliciousness_score = scores["maliciousness"]
        self.success = maliciousness_score >= self.threshold
        self.score = maliciousness_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Maliciousness"


class CorrectnessMetric(BaseMetric):
    """This metric checks the correctness using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        try:
            from ragas import evaluate
            from ragas.metrics.critique import correctness

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        correctness.llm = LangchainLLM(llm=chat_model)

        data = {
            "ground_truths": [[test_case.expected_output]],
            "contexts": [test_case.context],
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, metrics=[correctness])
        correctness_score = scores["correctness"]
        self.success = correctness_score >= self.threshold
        self.score = correctness_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Correctness"


class ConcisenessMetric(BaseMetric):
    """This metric checks the conciseness using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        try:
            from ragas import evaluate
            from ragas.metrics.critique import conciseness
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        chat_model = self.model.load_model()
        conciseness.llm = LangchainLLM(llm=chat_model)

        data = {
            "ground_truths": [[test_case.expected_output]],
            "contexts": [test_case.context],
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "id": [[test_case.id]],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, metrics=[conciseness])
        conciseness_score = scores["conciseness"]
        self.success = conciseness_score >= self.threshold
        self.score = conciseness_score
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Conciseness"


class RagasMetric(BaseMetric):
    """This metric checks if the output is more than 3 letters"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
    ):
        self.threshold = threshold
        self.model_name = model
        self.model = GPTModel(model=model)
        self.evaluation_model = self.model.get_model_name()

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            from ragas import evaluate
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            # How do i make sure this isn't just huggingface dataset
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Create a dataset from the test case
        # Convert the LLMTestCase to a format compatible with Dataset
        score_breakdown = {}
        metrics = [
            RAGASContextualPrecisionMetric(model=self.model_name),
            RAGASContextualRecallMetric(model=self.model_name),
            RAGASFaithfulnessMetric(model=self.model_name),
            RAGASAnswerRelevancyMetric(model=self.model_name),
        ]

        for metric in metrics:
            score = metric.measure(test_case)
            score_breakdown[metric.__name__] = score

        ragas_score = sum(score_breakdown.values()) / len(score_breakdown)

        self.success = ragas_score >= self.threshold
        self.score = ragas_score
        self.score_breakdown = score_breakdown
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "RAGAS"
