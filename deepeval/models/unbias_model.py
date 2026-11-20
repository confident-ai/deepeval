from typing import Optional
from deepeval.models.base_model import DeepEvalBaseModel


class UnBiasedModel(DeepEvalBaseModel):
    def __init__(self, model_name: str | None = None, *args, **kwargs):
        model_name = "original" if model_name is None else model_name
        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        # Imported lazily: `Dbias` is a dated package that must not break
        # plain `import deepeval`.
        # See https://github.com/confident-ai/deepeval/issues/382
        try:
            from Dbias.bias_classification import classifier
        except ImportError as e:
            raise ImportError(
                "UnBiasedModel needs the dated `Dbias` package, which is "
                "not installed. Run `pip install deepeval[bias]`."
            ) from e
        return classifier

    def _call(self, text):
        return self.model(text)
