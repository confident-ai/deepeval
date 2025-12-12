import warnings

from typing import Optional, Tuple, Union, Dict, List
from pydantic import BaseModel, SecretStr
import base64
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.utils import (
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.utils import require_dependency
from deepeval.models.llms.constants import ANTHROPIC_MODELS_DATA

# consistent retry rules
retry_anthropic = create_retry_decorator(PS.ANTHROPIC)

_ALIAS_MAP = {
    "api_key": ["_anthropic_api_key"],
}


class AnthropicModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "claude-3-7-sonnet-latest",
        api_key: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "AnthropicModel",
            kwargs,
            _ALIAS_MAP,
        )

        self.model_data = ANTHROPIC_MODELS_DATA.get(model)

        # re-map depricated keywords to re-named positional args
        if api_key is None and "api_key" in alias_values:
            api_key = alias_values["api_key"]

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = get_settings().ANTHROPIC_API_KEY

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    ###############################################
    # Generate functions
    ###############################################

    @retry_anthropic
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_payload_anthropic(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        chat_model = self.load_model()
        message = chat_model.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)
            return schema.model_validate(json_output), cost

    @retry_anthropic
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_payload_anthropic(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        chat_model = self.load_model(async_mode=True)
        message = await chat_model.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)

            return schema.model_validate(json_output), cost
        
    def generate_payload_anthropic(
        self, multimodal_input: List[Union[str, MLLMImage]]
    ):
        content = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                content.append({"type": "text", "text": ele})
            elif isinstance(ele, MLLMImage):
                image_format, image_raw_bytes = self._parse_image(ele)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_format}",
                            "data": base64.b64encode(image_raw_bytes).decode("utf-8"),
                        },
                    }
                )
        return content
    
    def _parse_image(self, image: MLLMImage):
        if image.dataBase64:
            fmt = (image.mimeType or "image/jpeg").split("/")[-1]
            try:
                raw_bytes = base64.b64decode(image.dataBase64)
            except Exception:
                raise ValueError("Invalid base64 in MLLMImage.dataBase64")
            return fmt, raw_bytes
        if image.local and image.filename:
            import PIL.Image
            from io import BytesIO
            img = PIL.Image.open(image.filename)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            fmt = (image.mimeType or "image/jpeg").split("/")[-1].lower()
            fmt = "jpeg" if fmt == "jpg" else fmt
            img.save(buf, format=fmt)
            raw_bytes = buf.getvalue()
            return fmt, raw_bytes
        if image.url:
            import requests
            resp = requests.get(image.url)
            resp.raise_for_status()
            raw_bytes = resp.content
            mime = resp.headers.get("content-type", image.mimeType or "image/jpeg")
            fmt = mime.split("/")[-1]
            return fmt, raw_bytes
        raise ValueError(
            "MLLMImage must contain dataBase64, or (local=True + filename), or url."
        )

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:

        if (
            self.model_data.input_price is None
            or self.model_data.output_price is None
        ):
            # Calculate average cost from all known models
            avg_input_cost = sum(
                model.input_price for model in ANTHROPIC_MODELS_DATA.values()
            ) / len(ANTHROPIC_MODELS_DATA)
            avg_output_cost = sum(
                model.output_price for model in ANTHROPIC_MODELS_DATA.values()
            ) / len(ANTHROPIC_MODELS_DATA)
            self.model_data.input_price = avg_input_cost
            self.model_data.output_price = avg_output_cost

            warnings.warn(
                f"[Warning] Pricing not defined for model '{self.name}'. "
                "Using average input/output token costs from existing model_pricing."
            )

        input_cost = input_tokens * self.model_data.input_price
        output_cost = output_tokens * self.model_data.output_price
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        module = require_dependency(
            "anthropic",
            provider_label="AnthropicModel",
            install_hint="Install it with `pip install anthropic`.",
        )

        if not async_mode:
            return self._build_client(module.Anthropic)
        return self._build_client(module.AsyncAnthropic)

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        # If we are managing retries with Tenacity, force SDK retries off to avoid double retries.
        # if the user opts into SDK retries via DEEPEVAL_SDK_RETRY_PROVIDERS, then honor their max_retries.
        if not sdk_retries_for(PS.ANTHROPIC):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Anthropic",
            env_var_name="ANTHROPIC_API_KEY",
            param_hint="`api_key` to AnthropicModel(...)",
        )
        kw = dict(
            api_key=api_key,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # in case older SDKs don’t accept max_retries, drop it and retry
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    def supports_multimodal(self):
        return self.model_data.supports_multimodal

    def get_model_name(self):
        return f"{self.name} (Anthropic)"
