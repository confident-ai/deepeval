from typing import Union

from deepeval.models import DeepEvalBaseEmbeddingModel, OpenAIEmbeddingModel


def initialize_embedding_model(
    model: Union[str, DeepEvalBaseEmbeddingModel] = None,
) -> Union[DeepEvalBaseEmbeddingModel, None]:
    if isinstance(model, str):
        return OpenAIEmbeddingModel(model=model)
    elif isinstance(model, DeepEvalBaseEmbeddingModel):
        return model
    else:
        # None
        return None
