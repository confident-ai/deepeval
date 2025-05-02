"""An implementation of the Ragas metric"""

from typing import Optional, Union, List


from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel
from deepeval.telemetry import capture_metric_type

# check langchain availability
try:
    import langchain_core
    from langchain_core.language_models import BaseChatModel
    from langchain_core.embeddings import Embeddings

    langchain_available = True
except ImportError:
    langchain_available = False


def _check_langchain_available():
    if not langchain_available:
        raise ImportError(
            "langchain_core is required for this functionality. Install it via your package manager"
        )


def format_ragas_metric_name(name: str):
    return f"{name} (ragas)"


#############################################################
# Context Precision
#############################################################


class RAGASContextualPrecisionMetric(BaseMetric):
    """This metric checks the contextual precision using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, "BaseChatModel"]] = "gpt-3.5-turbo",
        _track: bool = True,
    ):
        _check_langchain_available()
        self.threshold = threshold
        self.model = model
        self._track = _track
        if isinstance(model, str):
            self.evaluation_model = model

    def measure(self, test_case: LLMTestCase):
        try:
            import_ragas()
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
        if isinstance(self.model, str):
            chat_model = GPTModel(model=self.model).load_model()
        else:
            chat_model = self.model

        # Create a dataset from the test case
        data = {
            "contexts": [test_case.retrieval_context],
            "question": [test_case.input],
            "ground_truth": [test_case.expected_output],
        }
        dataset = Dataset.from_dict(data)

        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            # Evaluate the dataset using Ragas
            scores = evaluate(
                dataset, metrics=[context_precision], llm=chat_model
            )

            # Ragas only does dataset-level comparisons
            context_precision_score = scores["context_precision"][0]
            self.success = context_precision_score >= self.threshold
            self.score = context_precision_score
            return self.score

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Precision")


#############################################################
# Context Recall
#############################################################


class RAGASContextualRecallMetric(BaseMetric):
    """This metric checks the context recall using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, "BaseChatModel"]] = "gpt-3.5-turbo",
        _track: bool = True,
    ):
        self.threshold = threshold
        self.model = model
        self._track = _track
        if isinstance(model, str):
            self.evaluation_model = model

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ):
        return self.measure(test_case)

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
        if isinstance(self.model, str):
            chat_model = GPTModel(model=self.model).load_model()
        else:
            chat_model = self.model

        data = {
            "question": [test_case.input],
            "ground_truth": [test_case.expected_output],
            "contexts": [test_case.retrieval_context],
        }
        dataset = Dataset.from_dict(data)
        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            scores = evaluate(dataset, [context_recall], llm=chat_model)
            context_recall_score = scores["context_recall"][0]
            self.success = context_recall_score >= self.threshold
            self.score = context_recall_score
            return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Recall")


#############################################################
# Context Entities Recall
#############################################################


class RAGASContextualEntitiesRecall(BaseMetric):
    """This metric checks the context entities recall using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, "BaseChatModel"]] = "gpt-3.5-turbo",
        _track: bool = True,
    ):
        self.threshold = threshold
        self.model = model
        self._track = _track
        if isinstance(model, str):
            self.evaluation_model = model

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ):
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            import_ragas()
            from ragas import evaluate
            from ragas.metrics import ContextEntityRecall

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        if isinstance(self.model, str):
            chat_model = GPTModel(model=self.model).load_model()
        else:
            chat_model = self.model

        data = {
            "ground_truth": [test_case.expected_output],
            "contexts": [test_case.retrieval_context],
        }
        dataset = Dataset.from_dict(data)

        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            scores = evaluate(
                dataset,
                metrics=[ContextEntityRecall()],
                llm=chat_model,
            )
            contextual_entity_score = scores["context_entity_recall"][0]
            self.success = contextual_entity_score >= self.threshold
            self.score = contextual_entity_score
            return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Contextual Entities Recall")


#############################################################
# Noise Sensitivity
#############################################################

# class RAGASNoiseSensitivityMetric(BaseMetric):
#     """This metric checks the noise sensitivity using Ragas"""

#     def __init__(
#         self,
#         threshold: float = 0.3,
#         model: Optional[Union[str, BaseChatModel]] = "gpt-3.5-turbo",
#         _track: bool = True,
#     ):
#         self.threshold = threshold
#         self.model = model
#         self._track = _track
#         if isinstance(model, str):
#             self.evaluation_model = model

#     async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = False):
#         return self.measure(test_case)

#     def measure(self, test_case: LLMTestCase):
#         # sends to server
#         try:
#             import_ragas()
#             from ragas import evaluate
#             from ragas.metrics import NoiseSensitivity

#         except ModuleNotFoundError:
#             raise ModuleNotFoundError(
#                 "Please install ragas to use this metric. `pip install ragas`."
#             )

#         try:
#             from datasets import Dataset
#         except ModuleNotFoundError:
#             raise ModuleNotFoundError("Please install dataset")

#         # Set LLM model
#         if isinstance(self.model, str):
#             chat_model = GPTModel(model=self.model).load_model()
#         else:
#             chat_model = self.model

#         data = {
#             "question": [test_case.input],
#             "answer": [test_case.actual_output],
#             "ground_truth": [test_case.expected_output],
#             "contexts": [test_case.retrieval_context],
#         }
#         dataset = Dataset.from_dict(data)

#         with capture_metric_type(self.__name__, _track=self._track):
#             scores = evaluate(
#                 dataset,
#                 metrics=[NoiseSensitivity()],
#                 llm=chat_model,
#             )
#             noise_sensitivity_score = scores["noise_sensitivity_relevant"][0]
#             self.success = noise_sensitivity_score >= self.threshold
#             self.score = noise_sensitivity_score
#             return self.score

#     def is_successful(self):
#         return self.success

#     @property
#     def __name__(self):
#         return format_ragas_metric_name("Noise Sensitivity")


#############################################################
# Response Relevancy
#############################################################


class RAGASAnswerRelevancyMetric(BaseMetric):
    """This metric checks the answer relevancy using Ragas"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, "BaseChatModel"]] = "gpt-3.5-turbo",
        embeddings: Optional["Embeddings"] = None,
        _track: bool = True,
    ):
        self.threshold = threshold
        self.model = model
        self._track = _track
        if isinstance(model, str):
            self.evaluation_model = model
        self.embeddings = embeddings

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ):
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            import_ragas()
            from ragas import evaluate
            from ragas.metrics import ResponseRelevancy

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install ragas to use this metric. `pip install ragas`."
            )

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install dataset")

        # Set LLM model
        if isinstance(self.model, str):
            chat_model = GPTModel(model=self.model).load_model()
        else:
            chat_model = self.model

        data = {
            "question": [test_case.input],
            "answer": [test_case.actual_output],
            "contexts": [test_case.retrieval_context],
        }
        dataset = Dataset.from_dict(data)

        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            scores = evaluate(
                dataset,
                metrics=[ResponseRelevancy(embeddings=self.embeddings)],
                llm=chat_model,
                embeddings=self.embeddings,
            )
            answer_relevancy_score = scores["answer_relevancy"][0]
            self.success = answer_relevancy_score >= self.threshold
            self.score = answer_relevancy_score
            return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Answer Relevancy")


#############################################################
# Faithfulness
#############################################################


class RAGASFaithfulnessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, "BaseChatModel"]] = "gpt-3.5-turbo",
        _track: bool = True,
    ):
        self.threshold = threshold
        self.model = model
        self._track = _track
        if isinstance(model, str):
            self.evaluation_model = model

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ):
        return self.measure(test_case)

    def measure(self, test_case: LLMTestCase):
        # sends to server
        try:
            import_ragas()
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
        if isinstance(self.model, str):
            chat_model = GPTModel(model=self.model).load_model()
        else:
            chat_model = self.model

        data = {
            "contexts": [test_case.retrieval_context],
            "question": [test_case.input],
            "answer": [test_case.actual_output],
        }
        dataset = Dataset.from_dict(data)
        with capture_metric_type(
            self.__name__, _track=self._track, async_mode=False
        ):
            scores = evaluate(dataset, metrics=[faithfulness], llm=chat_model)
            faithfulness_score = scores["faithfulness"][0]
            self.success = faithfulness_score >= self.threshold
            self.score = faithfulness_score
            return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return format_ragas_metric_name("Faithfulness")


#############################################################
# RAGAS Metric
#############################################################


class RagasMetric(BaseMetric):
    """This metric checks if the output is more than 3 letters"""

    def __init__(
        self,
        threshold: float = 0.3,
        model: Optional[Union[str, "BaseChatModel"]] = "gpt-3.5-turbo",
        embeddings: Optional["Embeddings"] = None,
    ):
        self.threshold = threshold
        self.model = model
        if isinstance(model, str):
            self.evaluation_model = model
        self.embeddings = embeddings

    async def a_measure(
        self, test_case: LLMTestCase, _show_indicator: bool = False
    ):
        return self.measure(test_case)

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
        metrics: List[BaseMetric] = [
            RAGASContextualPrecisionMetric(model=self.model, _track=False),
            RAGASContextualRecallMetric(model=self.model, _track=False),
            RAGASContextualEntitiesRecall(model=self.model, _track=False),
            RAGASAnswerRelevancyMetric(
                model=self.model, embeddings=self.embeddings, _track=False
            ),
            RAGASFaithfulnessMetric(model=self.model, _track=False),
        ]

        with capture_metric_type(self.__name__, async_mode=False):
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


def import_ragas():
    import ragas

    required_version = "0.2.1"
    if not hasattr(ragas, "__version__"):
        raise ImportError("Version information is not available for ragas.")
    installed_version = ragas.__version__
    if installed_version < required_version:
        raise ImportError(
            f"ragas version {required_version} or higher is required, but {installed_version} is installed."
        )
