import os
from deepeval.models.base import DeepEvalBaseModel
from sentence_transformers import CrossEncoder
from deepeval.utils import softmax


class FactualConsistencyModel(DeepEvalBaseModel):
    def __init__(self, model_name: str | None = None, *args, **kwargs):
        model_name = (
            "cross-encoder/nli-deberta-v3-large"
            if model_name is None
            else model_name
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        return CrossEncoder(self.model_name)

    def _call(self, text_a: str, text_b: str):
        scores = self.model.predict([(text_a, text_b), (text_b, text_a)])
        # https://huggingface.co/cross-encoder/nli-deberta-base
        # label_mapping = ["contradiction", "entailment", "neutral"]
        softmax_scores = softmax(scores)
        score = softmax_scores[0][1]
        second_score = softmax_scores[1][1]
        return max(score, second_score)
