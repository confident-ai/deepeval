from typing import Optional, Tuple, Union, Dict
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, SecretStr
import base64
from deepeval.config.settings import get_settings
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import (
    require_secret_api_key,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.constants import KIMI_MODELS_DATA


retry_kimi = create_retry_decorator(PS.KIMI)


class KimiModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        model = model or settings.MOONSHOT_MODEL_NAME
        self.model_data = KIMI_MODELS_DATA.get(model)
        if model not in KIMI_MODELS_DATA.keys():
            raise ValueError(
                f"Invalid model. Available Moonshot models: {', '.join(KIMI_MODELS_DATA.keys())}"
            )

        temperature_from_key = settings.TEMPERATURE
        if temperature_from_key is None:
            self.temperature = temperature
        else:
            self.temperature = float(temperature_from_key)
        if self.temperature < 0:
            raise ValueError("Temperature must be >= 0.")

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = settings.MOONSHOT_API_KEY

        self.base_url = "https://api.moonshot.cn/v1"
        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_kimi
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_payload_kimi(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        client = self.load_model(async_mode=False)
        if schema and self.model_data.supports_json:
            completion = client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                **self.generation_kwargs,
            )
            json_output = trim_and_load_json(
                completion.choices[0].message.content
            )
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            return schema.model_validate(json_output), cost

        completion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": content}],
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    @retry_kimi
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_payload_kimi(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        client = self.load_model(async_mode=True)
        if schema and self.model_data.supports_json:
            completion = await client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                **self.generation_kwargs,
            )
            json_output = trim_and_load_json(
                completion.choices[0].message.content
            )
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            return schema.model_validate(json_output), cost

        completion = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": content}],
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost
        
    def generate_payload_kimi(self, multimodal_input):
        """
        Converts multimodal input into Kimi/OpenAI-compatible messages.
        """
        content = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                content.append({"type": "text", "text": ele})
            elif isinstance(ele, MLLMImage):
                mime, raw_bytes = self._parse_image(ele)
                b64 = base64.b64encode(raw_bytes).decode("utf-8")
                content.append({"type": "image", "image_url": f"data:{mime};base64,{b64}"})
        return content
    
    def _parse_image(self, image: MLLMImage):
        if image.dataBase64:
            mime = image.mimeType or "image/jpeg"
            return mime, base64.b64decode(image.dataBase64)

        if image.local and image.filename:
            from PIL import Image
            from io import BytesIO
            img = Image.open(image.filename)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            fmt = (image.mimeType or "image/jpeg").split("/")[-1].lower()
            fmt = "jpeg" if fmt == "jpg" else fmt
            img.save(buf, format=fmt.upper())
            return f"image/{fmt}", buf.getvalue()

        if image.url:
            import requests
            resp = requests.get(image.url)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", image.mimeType or "image/jpeg")
            return mime, resp.content

        raise ValueError(
            "MLLMImage must contain dataBase64, or (local=True + filename), or url."
        )


    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        input_cost = input_tokens * self.model_data.input_price
        output_cost = output_tokens * self.model_data.output_price
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, force OpenAI SDK retries off to avoid double retries.
        If the user opts into SDK retries for 'kimi' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave their retry settings as is.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.KIMI):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Kimi",
            env_var_name="MOONSHOT_API_KEY",
            param_hint="`api_key` to KimiModel(...)",
        )

        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    def get_model_name(self):
        return f"{self.name} (KIMI)"
    
    def supports_multimodal(self):
        return self.model_data.supports_multimodal
