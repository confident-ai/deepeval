from deepeval.models.base_model import DeepEvalBaseModel


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
        # Imported lazily: `detoxify` (and its torch dependency) are dated,
        # heavy packages that must not break plain `import deepeval`.
        # See https://github.com/confident-ai/deepeval/issues/382
        try:
            import torch
            from detoxify import Detoxify
        except ImportError as e:
            raise ImportError(
                "DetoxifyModel needs the dated `detoxify` package, which is "
                "not installed. Run `pip install deepeval[toxicity]`."
            ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return Detoxify(self.model_name, device=device)

    def _call(self, text: str):
        toxicity_score_dict = self.model.predict(text)
        mean_toxicity_score = sum(list(toxicity_score_dict.values())) / len(
            toxicity_score_dict
        )
        return mean_toxicity_score, toxicity_score_dict
