from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from typing import Optional, List

from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from openai import OpenAI

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
            if (
                not self.should_use_local_embeddings()
                and model_name not in valid_openai_embedding_models
            ):
                raise ValueError(
                    f"Invalid model. Available OpenAI Embedding models: {', '.join(model for model in valid_openai_embedding_models)}"
                )
        elif model is None:
            model_name = default_openai_embedding_model
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        if self.should_use_azure_openai():
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_API_KEY
            )

            openai_api_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.OPENAI_API_VERSION
            )
            azure_embedding_deployment = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME
            )
            azure_endpoint = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_ENDPOINT
            )

            return AzureOpenAIEmbeddings(
                openai_api_version=openai_api_version,
                azure_deployment=azure_embedding_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                **self.kwargs,
            )
        elif self.should_use_local_embeddings():
            base_url = KEY_FILE_HANDLER.fetch_data(
                KeyValues.LOCAL_EMBEDDING_BASE_URL
            )
            api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.LOCAL_EMBEDDING_API_KEY
            )
            self.model_name = KEY_FILE_HANDLER.fetch_data(
                KeyValues.LOCAL_EMBEDDING_MODEL_NAME
            )
            if self.should_use_local_ollama():
                return OpenAI(base_url=base_url, api_key=api_key)
            else:
                return OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_base=base_url,
                    openai_api_key=api_key,
                    **self.kwargs,
                )
        else:
            return OpenAIEmbeddings(model=self.model_name, **self.kwargs)

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        if self.should_use_local_ollama():
            response = embedding_model.embeddings.create(
                model=self.model_name,
                input=[text],
            )
            return response.data[0].embedding
        else:
            return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        if self.should_use_local_ollama():
            response = embedding_model.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [data.embedding for data in response.data]
        else:
            return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        if self.should_use_local_ollama():
            response = embedding_model.embeddings.create(
                model=self.model_name,
                input=[text],
            )
            return response.data[0].embedding
        else:
            return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        if self.should_use_local_ollama():
            response = embedding_model.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [data.embedding for data in response.data]
        else:
            return await embedding_model.aembed_documents(texts)

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def should_use_local_embeddings(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_LOCAL_EMBEDDINGS)
        return value.lower() == "yes" if value is not None else False

    def should_use_local_ollama(self):
        base_url = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_EMBEDDING_BASE_URL
        )
        return base_url == "http://localhost:11434/v1/"

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.should_use_local_embeddings():
            return "local embeddings"
        elif self.model_name:
            return self.model_name
