from typing import Optional, Union

from deepeval.models.base_model import (
    DeepEvalBaseLLM,
    DeepEvalBaseEmbeddingModel,
)


class BaseSynthesizer:
    synthesizer_model: Optional[str] = None
    embedding_model: Optional[str] = None

    @property
    def model(self) -> float:
        return self._model

    @model.setter
    def model(self, model: Optional[Union[str, DeepEvalBaseLLM]] = None):
        self._model = model

    @property
    def embedder(self) -> float:
        return self._embedder

    @embedder.setter
    def embedder(
        self, embedder: Optional[Union[str, DeepEvalBaseEmbeddingModel]] = None
    ):
        self._embedder = embedder
