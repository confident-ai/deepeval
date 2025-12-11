import base64
from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional, Tuple, Union, Dict, List
from deepeval.test_case import MLLMImage
from pydantic import BaseModel, SecretStr
from io import BytesIO
from openai import (
    OpenAI,
    AsyncOpenAI,
)
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import (
    parse_model_name,
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.constants import OPENAI_MODELS_DATA


retry_openai = create_retry_decorator(PS.OPENAI)

default_gpt_model = "gpt-4.1"


def _request_timeout_seconds() -> float:
    timeout = float(get_settings().DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0)
    return timeout if timeout > 0 else 30.0


_ALIAS_MAP = {
    "api_key": ["_openai_api_key"],
}


class GPTModel(DeepEvalBaseLLM):

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "GPTModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if api_key is None and "api_key" in alias_values:
            api_key = alias_values["api_key"]

        settings = get_settings()
        model = model or settings.OPENAI_MODEL_NAME
        if model is None:
            model = default_gpt_model
        self.model_data = OPENAI_MODELS_DATA.get(model)
        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.OPENAI_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.OPENAI_COST_PER_OUTPUT_TOKEN
        )

        if isinstance(model, str):
            model = parse_model_name(model)
            if model not in OPENAI_MODELS_DATA.keys():
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in OPENAI_MODELS_DATA.keys())}"
                )

        if (
            self.model_data.input_price is None
            or self.model_data.output_price is None
        ):
            if cost_per_input_token is None or cost_per_output_token is None:
                raise ValueError(
                    f"No pricing available for `{model}`. "
                    "Please provide both `cost_per_input_token` and `cost_per_output_token` when initializing `GPTModel`, "
                    "or set them via the CLI:\n"
                    "    deepeval set-openai --model=[...] --cost_per_input_token=[...] --cost_per_output_token=[...]"
                )
            else:
                self.model_data.input_price = (float(cost_per_input_token))
                self.model_data.output_price = (float(cost_per_output_token))

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = get_settings().OPENAI_API_KEY

        self.base_url = base_url
        # args and kwargs will be passed to the underlying model, in load_model function

        if not self.model_data.supports_temperature:
            temperature = 1

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

    @retry_openai
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            prompt = self.generate_prompt(prompt)

        if schema:
            if self.model_data.supports_structured_outputs:
                completion = client.beta.chat.completions.parse(
                    model=self.name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format=schema,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost
            if self.model_data.supports_json:
                completion = client.beta.chat.completions.parse(
                    model=self.name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
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
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    @retry_openai
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            prompt = self.generate_prompt(prompt)

        if schema:
            if self.model_data.supports_structured_outputs:
                completion = await client.beta.chat.completions.parse(
                    model=self.name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format=schema,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost
            if self.model_data.supports_json:
                completion = await client.beta.chat.completions.parse(
                    model=self.name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
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
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    ###############################################
    # Other generate functions
    ###############################################

    @retry_openai
    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=False)
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            prompt = self.generate_prompt(prompt)
        completion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)

        return completion, cost

    @retry_openai
    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=True)
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            prompt = self.generate_prompt(prompt)
        completion = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)

        return completion, cost

    @retry_openai
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[list[str], float]:
        client = self.load_model(async_mode=False)
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            prompt = self.generate_prompt(prompt)
        response = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            **self.generation_kwargs,
        )
        completions = [choice.message.content for choice in response.choices]
        return completions

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # TODO: consider loggin a warning instead of defaulting to whole model pricing
        input_cost = input_tokens * self.model_data.input_price
        output_cost = output_tokens * self.model_data.input_price
        return input_cost + output_cost

    #########
    # Model #
    #########

    def generate_prompt(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ):
        prompt = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                prompt.append({"type": "text", "text": ele})
            elif isinstance(ele, MLLMImage):
                if ele.local:
                    import PIL.Image

                    image = PIL.Image.open(ele.url)
                    visual_dict = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.encode_pil_image(image)}"
                        },
                    }
                else:
                    visual_dict = {
                        "type": "image_url",
                        "image_url": {"url": ele.url},
                    }
                prompt.append(visual_dict)
        return prompt

    def encode_pil_image(self, pil_image):
        image_buffer = BytesIO()
        if pil_image.mode in ("RGBA", "LA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return base64_encoded_image

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, force OpenAI SDK retries off to avoid double retries.
        If the user opts into SDK retries for 'openai' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave their retry settings as is.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.OPENAI):
            kwargs["max_retries"] = 0

        if not kwargs.get("timeout"):
            kwargs["timeout"] = _request_timeout_seconds()
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenAI",
            env_var_name="OPENAI_API_KEY",
            param_hint="`api_key` to GPTModel(...)",
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

    def supports_multimodal(self):
        return self.model_data.supports_multimodal

    def get_model_name(self):
        return f"{self.name}"
