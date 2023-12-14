from typing import Dict, Optional

from langchain.chat_models import ChatOpenAI
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

        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        return ChatOpenAI(
            model_name=self.model_name, model_kwargs=self.model_kwargs
        )

    def _call(self, prompt: str):
        chat_model = self.load_model()
        return call_openai_with_retry(lambda: chat_model.invoke(prompt))
