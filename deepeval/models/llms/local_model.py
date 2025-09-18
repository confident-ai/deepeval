from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from deepeval.constants import ProviderSlug as PS


# consistent retry rules
retry_local = create_retry_decorator(PS.LOCAL)


class LocalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0,
        format: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.LOCAL_MODEL_NAME
        )
        self.local_model_api_key = api_key or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.LOCAL_MODEL_API_KEY
        )
        self.base_url = base_url or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.LOCAL_MODEL_BASE_URL
        )
        self.format = format or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.LOCAL_MODEL_FORMAT
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_local
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        response: ChatCompletion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    @retry_local
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=True)
        response: ChatCompletion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        model_name = KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.LOCAL_MODEL_NAME
        )
        return f"{model_name} (Local Model)"

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity manages retries, turn off OpenAI SDK retries to avoid double retrying.
        If users opt into SDK retries via DEEPEVAL_SDK_RETRY_PROVIDERS=local, leave them enabled.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.LOCAL):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        kw = dict(
            api_key=self.local_model_api_key,
            base_url=self.base_url,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # Older OpenAI SDKs may not accept max_retries; drop and retry once.
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
