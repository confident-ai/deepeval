from typing import Optional, Tuple, Union, Dict, List
from pydantic import BaseModel, SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.llms.multimodal import reject_audio, to_openai_content
from deepeval.test_case import Audio
from deepeval.models.utils import (
    require_costs,
    require_secret_api_key,
    EvaluationCost,
    token_cost,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.constants import GROK_MODELS_DATA
from deepeval.utils import require_param

# consistent retry rules
retry_grok = create_retry_decorator(PS.GROK)


class GrokModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):

        settings = get_settings()

        model = model or settings.GROK_MODEL_NAME

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.GROK_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.GROK_COST_PER_OUTPUT_TOKEN
        )

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.GROK_API_KEY

        model = require_param(
            model,
            provider_label="GrokModel",
            env_var_name="GROK_MODEL_NAME",
            param_hint="model",
        )

        # validation
        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")

        self.model_data = GROK_MODELS_DATA.get(model)
        self.temperature = temperature

        cost_per_input_token, cost_per_output_token = require_costs(
            self.model_data,
            model,
            "GROK_COST_PER_INPUT_TOKEN",
            "GROK_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token,
            cost_per_output_token,
        )
        self.model_data.input_price = cost_per_input_token
        self.model_data.output_price = cost_per_output_token

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        super().__init__(model)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_grok
    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Union[str, BaseModel], float]:

        try:
            from xai_sdk.chat import user
        except ImportError:
            raise ImportError(
                "xai_sdk is required to use GrokModel. Please install it with: pip install xai-sdk"
            )
        content = self._prompt_content(prompt, audio)

        client = self.load_model(async_mode=False)
        chat = client.chat.create(
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        chat.append(user(content))

        if schema and self.supports_structured_outputs() is True:
            response, structured_output = chat.parse(schema)
            cost = self.calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            return structured_output, cost

        response = chat.sample()
        output = response.content
        cost = self.calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    @retry_grok
    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Union[str, BaseModel], float]:

        try:
            from xai_sdk.chat import user
        except ImportError:
            raise ImportError(
                "xai_sdk is required to use GrokModel. Please install it with: pip install xai-sdk"
            )

        content = self._prompt_content(prompt, audio)

        client = self.load_model(async_mode=True)
        chat = client.chat.create(
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        chat.append(user(content))

        if schema and self.supports_structured_outputs() is True:
            response, structured_output = await chat.parse(schema)
            cost = self.calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            return structured_output, cost

        response = await chat.sample()
        output = response.content
        cost = self.calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    def _prompt_content(
        self, prompt: str, audio: Optional[List[Audio]] = None
    ) -> List[Dict]:
        # xAI's chat API takes text and images, with no audio input.
        reject_audio(audio, "GrokModel")
        return to_openai_content(prompt)

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> Optional[EvaluationCost]:
        return token_cost(self.model_data, input_tokens, output_tokens)

    ###############################################
    # Capabilities
    ###############################################

    def supports_log_probs(self) -> Union[bool, None]:
        return self.model_data.supports_log_probs

    def supports_temperature(self) -> Union[bool, None]:
        return self.model_data.supports_temperature

    def supports_multimodal(self) -> Union[bool, None]:
        return self.model_data.supports_multimodal

    def supports_audio(self) -> bool:
        return False

    def supports_structured_outputs(self) -> Union[bool, None]:
        return self.model_data.supports_structured_outputs

    def supports_json_mode(self) -> Union[bool, None]:
        return self.model_data.supports_json

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        try:
            from xai_sdk import Client, AsyncClient

            if not async_mode:
                return self._build_client(Client)
            else:
                return self._build_client(AsyncClient)
        except ImportError:
            raise ImportError(
                "xai_sdk is required to use GrokModel. Please install it with: pip install xai-sdk"
            )

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, disable gRPC channel retries to avoid double retry.
        If the user opts into SDK retries for 'grok' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave channel options as is
        """
        kwargs = dict(self.kwargs or {})
        opts = list(kwargs.get("channel_options", []))
        if not sdk_retries_for(PS.GROK):
            # remove any explicit enable flag, then disable retries
            opts = [
                option
                for option in opts
                if not (
                    isinstance(option, (tuple, list))
                    and option
                    and option[0] == "grpc.enable_retries"
                )
            ]
            opts.append(("grpc.enable_retries", 0))
        if opts:
            kwargs["channel_options"] = opts
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Grok",
            env_var_name="GROK_API_KEY",
            param_hint="`api_key` to GrokModel(...)",
        )

        kw = dict(api_key=api_key, **self._client_kwargs())
        try:
            return cls(**kw)
        except TypeError as e:
            # fallback: older SDK version might not accept channel_options
            if "channel_options" in str(e):
                kw.pop("channel_options", None)
                return cls(**kw)
            raise

    def get_model_name(self):
        return f"{self.name} (Grok)"
