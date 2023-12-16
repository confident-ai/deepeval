import os 
import numpy as np
from deepeval.models.base import DeepEvalBaseModel
from transformers import pipeline

class NERModel(DeepEvalBaseModel):
    def __init__(self, model_name: str | None = None, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "huggingface-course/bert-finetuned-ner"
        return pipeline(task="token-classification", model=model_name)

    def _call(self, text: str):
        ner_score_dict = self.model(text)
        entity = []
        for element in ner_score_dict:
            entity.append(element['entity'])
        return np.array(entity)
