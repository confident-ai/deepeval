import os
from typing import Optional
from deepeval.singleton import Singleton
from deepeval.progress_context import progress_context


class HallucinationModel(metaclass=Singleton):
    def __init__(self, model_name: Optional[str] = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "The 'sentence_transformers' library is required to use the HallucinationMetric."
            )
        # We use a smple cross encoder model
        model_name = (
            "vectara/hallucination_evaluation_model"
            if model_name is None
            else model_name
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # TODO: add this progress context in the correct place
        with progress_context(
            "Downloading HallucinationEvaluationModel (may take up to 2 minutes if running for the first time)..."
        ):
            self.model = CrossEncoder(model_name)
