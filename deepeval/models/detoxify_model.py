import torch
from deepeval.models.base_model import DeepEvalBaseModel
from detoxify import Detoxify


class DetoxifyModel(DeepEvalBaseModel):
    def __init__(self, model_name: str | None = None, *args, **kwargs):
        if model_name is not None:
            assert model_name in [
                "original",
                "unbiased",
                "multilingual",
            ], "Invalid model. Available variants: original, unbiased, multilingual"
        model_name = "original" if model_name is None else model_name
        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return Detoxify(self.model_name, device=device)

    def _call(self, text: str):
        toxicity_score_dict = self.model.predict(text)
        mean_toxicity_score = sum(list(toxicity_score_dict.values())) / len(
            toxicity_score_dict
        )
        return mean_toxicity_score, toxicity_score_dict
