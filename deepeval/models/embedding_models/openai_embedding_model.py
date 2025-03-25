from langchain_openai import OpenAIEmbeddings
from typing import Optional, List

from deepeval.models import DeepEvalBaseEmbeddingModel

valid_openai_embedding_models = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
default_openai_embedding_model = "text-embedding-3-small"


class OpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_openai_embedding_models:
                raise ValueError(
                    f"Invalid model. Available OpenAI Embedding models: {', '.join(model for model in valid_openai_embedding_models)}"
                )
        elif model is None:
            model_name = default_openai_embedding_model
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        return OpenAIEmbeddings(model=self.model_name, **self.kwargs)

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def get_model_name(self):
        return self.model_name
