from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel, SecretStr
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from deepeval.config.settings import get_settings
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import require_secret_api_key
from deepeval.models import DeepEvalBaseLLM
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
        settings = get_settings()

        model_name = model or settings.LOCAL_MODEL_NAME
        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.local_model_api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.local_model_api_key = settings.LOCAL_MODEL_API_KEY

        self.base_url = (
            base_url
            or settings.LOCAL_MODEL_BASE_URL
            and str(settings.LOCAL_MODEL_BASE_URL)
        )
        self.format = format or settings.LOCAL_MODEL_FORMAT
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
        return f"{self.model_name} (Local Model)"

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
        local_model_api_key = require_secret_api_key(
            self.local_model_api_key,
            provider_label="Local",
            env_var_name="LOCAL_MODEL_API_KEY",
            param_hint="`api_key` to LocalModel(...)",
        )

        kw = dict(
            api_key=local_model_api_key,
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
