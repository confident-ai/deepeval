
import os
from deepeval.singleton import Singleton
from deepeval.test_case import LLMTestCase
from deepeval.utils import chunk_text, softmax
from deepeval.metrics.base_metric import BaseMetric
from deepeval.evaluator import assert_test
from deepeval.progress_context import progress_context
from sentence_transformers import CrossEncoder

class VectaraHallucinationEvaluationModel(metaclass=Singleton):
    def __init__(self, model_name: str = "vectara/hallucination_evaluation_model"):
        # We use a smple cross encoder model
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = CrossEncoder(model_name)

    def predict(self, text_a: str, text_b: str):
        scores = self.model.predict([[text_a, text_b]])
        # https://huggingface.co/vectara/hallucination_evaluation_model
        
        return scores[0]

class HallucinationEvaluationMetric(BaseMetric, metaclass=Singleton):
    def __init__(self, 
        minimum_score: float = 0.5,
        model_name: str = "vectara/hallucination_evaluation_model",
    ):
        # For Crossencoder model, move to singleton to avoid re-instantiating

        with progress_context(
            "Downloading HallucinationEvaluationModel (may take up to 2 minutes if running for the first time)..."
        ):
            self.model = VectaraHallucinationEvaluationModel(model_name)
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase):
        if test_case.actual_output is None or test_case.context is None:
            raise ValueError("Output or context cannot be None")

        context_list = []
        if isinstance(test_case.context, str):
            context_list.extend(chunk_text(test_case.context))
        elif isinstance(test_case.context, list):
            for context in test_case.context:
                context_list.extend(chunk_text(context))
        else:
            raise ValueError("Context must be a string or a list of strings")

        max_score = 0
        for c in context_list:
            score = self.model.predict(c, test_case.actual_output)
            if score > max_score:
                max_score = score

        self.success = max_score > self.minimum_score
        self.score = max_score
        return max_score
    
    def is_successful(self) -> bool:
        return self.success    