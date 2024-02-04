import os 
import numpy as np
from deepeval.models.base import DeepEvalBaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class NERModel(DeepEvalBaseModel):
    def __init__(self, model_name: str | None = None, *args, **kwargs):
        model_name = "dslim/bert-base-NER" if model_name is None else model_name
        super().__init__(model_name=model_name, *args, **kwargs)
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

    def load_model(self):
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        return model


    def _call(self, text: str):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(text, return_tensors="pt")
        nlp = pipeline("ner",model = self.model,tokenizer = self.tokenizer)
        outputs = nlp(text)
        return outputs
