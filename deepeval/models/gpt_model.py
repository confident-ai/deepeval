import os
from typing import Dict, Optional

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models.base import DeepEvalBaseModel
from deepeval.chat_completion.retry import call_openai_with_retry

valid_gpt_models = [
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
]

default_gpt_model = "gpt-4-1106-preview"


class GPTModel(DeepEvalBaseModel):
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_kwargs: Dict = {},
        *args,
        **kwargs,
    ):
        if model_name is not None:
            assert (
                model_name in valid_gpt_models
            ), f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
        else:
            model_name = default_gpt_model

        self.model_kwargs = model_kwargs

        # TODO: you should set the self.model here instead of loading it everytime

        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        if self.should_use_azure_openai():
            model_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_MODEL_VERSION
            )
            model_kwargs = {}
            if model_version is not None:
                model_kwargs["model_version"] = model_version

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
            return AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_kwargs=model_kwargs,
            )
        return ChatOpenAI(
            model_name=self.model_name, model_kwargs=self.model_kwargs
        )

    def _call(self, prompt: str):
        chat_model = self.load_model()
        return call_openai_with_retry(lambda: chat_model.invoke(prompt))

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False
