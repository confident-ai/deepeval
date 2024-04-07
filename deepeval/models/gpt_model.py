from typing import Optional, Tuple
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import AIMessage

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseLLM
from deepeval.chat_completion.retry import retry_with_exponential_backoff

valid_gpt_models = [
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0125",
]

default_gpt_model = "gpt-4-0125-preview"


class GPTModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_gpt_models:
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
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

            return AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_version=model_version,
            )

        return ChatOpenAI(
            model_name=self.model_name, openai_api_key=self._openai_api_key
        )

    @retry_with_exponential_backoff
    def generate(self, prompt: str) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res.content, cb.total_cost

    @retry_with_exponential_backoff
    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
            return res.content, cb.total_cost

    @retry_with_exponential_backoff
    def generate_raw_response(
        self, prompt: str, **kwargs
    ) -> Tuple[AIMessage, float]:
        chat_model = self.load_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res, cb.total_cost

    @retry_with_exponential_backoff
    async def a_generate_raw_response(
        self, prompt: str, **kwargs
    ) -> Tuple[AIMessage, float]:
        chat_model = self.load_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
        return res, cb.total_cost

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.model_name:
            return self.model_name
