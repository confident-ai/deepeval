from typing import Union

from deepeval.models import DeepEvalBaseEmbeddingModel, OpenAIEmbeddingModel


def initialize_embedding_model(
    model: Union[str, DeepEvalBaseEmbeddingModel] = None,
) -> DeepEvalBaseEmbeddingModel:
    if isinstance(model, DeepEvalBaseEmbeddingModel):
        return model
    elif isinstance(model, str):
        return OpenAIEmbeddingModel(model=model)
    else:
        return OpenAIEmbeddingModel()
