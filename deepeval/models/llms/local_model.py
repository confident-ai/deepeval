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
from deepeval.models.utils import (
    require_secret_api_key,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from deepeval.test_case import MLLMImage
from deepeval.utils import (
    check_if_multimodal,
    convert_to_multi_modal_array,
    require_param,
)


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
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        client = self.load_model(async_mode=False)
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
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        client = self.load_model(async_mode=True)
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

    def generate_content(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ):
        """
        Converts multimodal input into OpenAI-compatible format.
        Uses data URIs for all images since we can't guarantee local servers support URL fetching.
        """
        prompt = []
        for element in multimodal_input:
            if isinstance(element, str):
                prompt.append({"type": "text", "text": element})
            elif isinstance(element, MLLMImage):
                # For local servers, use data URIs for both remote and local images
                # Most local servers don't support fetching external URLs
                if element.url and not element.local:
                    import requests
                    import base64

                    settings = get_settings()
                    try:
                        response = requests.get(
                            element.url,
                            timeout=(
                                settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
                                settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
                            ),
                        )
                        response.raise_for_status()

                        # Get mime type from response
                        mime_type = response.headers.get(
                            "content-type", element.mimeType or "image/jpeg"
                        )

                        # Encode to base64
                        b64_data = base64.b64encode(response.content).decode(
                            "utf-8"
                        )
                        data_uri = f"data:{mime_type};base64,{b64_data}"

                    except Exception as e:
                        raise ValueError(
                            f"Failed to fetch remote image {element.url}: {e}"
                        )
                else:
                    element.ensure_images_loaded()
                    mime_type = element.mimeType or "image/jpeg"
                    data_uri = f"data:{mime_type};base64,{element.dataBase64}"

                prompt.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )
        return prompt

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
