import contextlib
from typing import Optional, Tuple, Union, Dict, List
from pydantic import BaseModel, SecretStr
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.llms.multimodal import to_openai_content
from deepeval.test_case import Audio
from deepeval.models.utils import (
    require_secret_api_key,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from deepeval.utils import require_param

# consistent retry rules
retry_local = create_retry_decorator(PS.LOCAL)


class LocalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        format: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        model = model or settings.LOCAL_MODEL_NAME
        if api_key is not None:
            self.local_model_api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.local_model_api_key = settings.LOCAL_MODEL_API_KEY

        base_url = (
            base_url if base_url is not None else settings.LOCAL_MODEL_BASE_URL
        )
        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else None
        )
        self.format = format or settings.LOCAL_MODEL_FORMAT or "json"

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        # validation
        model = require_param(
            model,
            provider_label="LocalModel",
            env_var_name="LOCAL_MODEL_NAME",
            param_hint="model",
        )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")
        self.temperature = temperature

        self.kwargs = kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        super().__init__(model)

    ###############################################
    # Generate functions
    ###############################################

    @retry_local
    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Union[str, BaseModel], float]:

        content = to_openai_content(
            prompt,
            audio,
            text_only_as_string=True,
            inline_remote_images=True,
        )

        with self._scoped_client() as client:
            response: ChatCompletion = client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": content}],
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
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Union[str, BaseModel], float]:

        content = to_openai_content(
            prompt,
            audio,
            text_only_as_string=True,
            inline_remote_images=True,
        )

        async with self._scoped_async_client() as client:
            response: ChatCompletion = await client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": content}],
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
        return f"{self.name} (Local Model)"

    def supports_multimodal(self):
        return True

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _owns_http_client(self) -> bool:
        """Whether clients built by this model own their HTTP transport.

        When the user passes ``http_client`` through ``kwargs``, every client
        this model builds wraps that same object, and ``close()`` would shut
        it down for all later calls, so its lifetime stays with the user.
        """
        return (self.kwargs or {}).get("http_client") is None

    @contextlib.contextmanager
    def _scoped_client(self):
        """Yield a fresh ``OpenAI`` client scoped to a single call.

        A new client is built per call, so it is closed again on exit instead
        of being left to the garbage collector.
        """
        client = self.load_model(async_mode=False)
        try:
            yield client
        finally:
            if self._owns_http_client():
                client.close()

    @contextlib.asynccontextmanager
    async def _scoped_async_client(self):
        """Yield a fresh ``AsyncOpenAI`` client scoped to a single call.

        The client is closed in the event loop that created it. Leaving it to
        the garbage collector leaks connections and, once that loop is gone,
        can print ``RuntimeError: Event loop is closed`` tracebacks.
        """
        client = self.load_model(async_mode=True)
        try:
            yield client
        finally:
            if self._owns_http_client():
                await client.close()

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
