from typing import Optional, List
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from deepeval.chat_completion.retry import retry_with_exponential_backoff

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
        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        if self.should_use_azure_openai():
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_API_KEY
            )

            openai_api_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.OPENAI_API_VERSION
            )
            azure_deployment = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_DEPLOYMENT_NAME
            )
            azure_endpoint = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_ENDPOINT
            )

            model_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_MODEL_VERSION
            )

            if model_version is None:
                model_version = ""

            return AzureOpenAIEmbeddings(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_version=model_version,
            )

        return OpenAIEmbeddings(model=self.model_name)

    def embed_query(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.model_name:
            return self.model_name


############################
###### Example Usage #######
############################

"""
import time
import asyncio

async def async_main(model):
    start_async = time.time()
    async_result = await model.aembed_query('test')
    end_async = time.time()
    
    print(f"Asynchronous Execution time: {end_async - start_async} seconds")

def main():
    model = OpenAIEmbeddingModel()

    start_sync = time.time()
    sync_result = model.embed_query('test')
    end_sync = time.time()

    print(f"Synchronous Execution time: {end_sync - start_sync} seconds")

    # Call the asynchronous part using asyncio.run
    asyncio.run(async_main(model))

if __name__ == "__main__":
    main()
"""
