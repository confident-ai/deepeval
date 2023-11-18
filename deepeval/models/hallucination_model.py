import os
from deepeval.singleton import Singleton
from sentence_transformers import CrossEncoder
from deepeval.models.model_map import model_map, name_to_card


class HallucinationModel(metaclass=Singleton):
    def __init__(self, model_name: str = "vectara-hallucination"):
        # We use a smple cross encoder model
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model_name = name_to_card(model_name)
        self.model = CrossEncoder(model_name)

    def score(self, source: str, prediction: str):
        scores = self.model.predict([[source, prediction]])
        # https://huggingface.co/vectara/hallucination_evaluation_model

        return scores[0]
