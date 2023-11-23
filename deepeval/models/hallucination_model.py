import os
from deepeval.singleton import Singleton
from sentence_transformers import CrossEncoder
from deepeval.progress_context import progress_context

class HallucinationModel(metaclass=Singleton):
    def __init__(self, model_name: str = "vectara/hallucination_evaluation_model"):
        # We use a smple cross encoder model
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # TODO: add this progress context in the correct place
        with progress_context(
            "Downloading HallucinationEvaluationModel (may take up to 2 minutes if running for the first time)..."
        ):
            self.model = CrossEncoder(model_name)
