from tenacity import retry, retry_if_exception_type, wait_exponential_jitter
from openai import OpenAI, AsyncOpenAI
from typing import Optional, Tuple
from pydantic import BaseModel
import logging
import openai

from deepeval.models import DeepEvalBaseLLM


def log_retry_error(retry_state):
    logging.error(
        f"OpenAI rate limit exceeded. Retrying: {retry_state.attempt_number} time(s)..."
    )


valid_gpt_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o3-mini",
    "o3-mini-2025-01-31",
]

structured_outputs_models = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o3-mini",
    "o3-mini-2025-01-31",
]

json_mode_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-32k",
    "gpt-4-32k-0613",
]

default_gpt_model = "gpt-4o"


class SchematicGPTModel(DeepEvalBaseLLM):
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
            if model_name in structured_outputs_models:
                self.structured_output = True
        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        self.is_azure_model: bool
        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        pass

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        import instructor

        client = instructor.from_openai(OpenAI(api_key=self._openai_api_key))
        response = client.chat.completions.create(
            model=self.model_name,
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
        )
        return response

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        import instructor

        client = instructor.from_openai(
            AsyncOpenAI(api_key=self._openai_api_key)
        )
        response = await client.chat.completions.create(
            model=self.model_name,
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
        )
        return response

    def get_model_name(self):
        return self.model_name
