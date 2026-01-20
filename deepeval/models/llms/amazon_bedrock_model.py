import base64
from typing import Optional, Tuple, Union, Dict, List
from contextlib import AsyncExitStack

from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings
from deepeval.utils import (
    require_dependency,
    require_param,
)
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.test_case import MLLMImage
from deepeval.errors import DeepEvalError
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.constants import BEDROCK_MODELS_DATA
from deepeval.models.llms.utils import trim_and_load_json, safe_asyncio_run
from deepeval.constants import ProviderSlug as PS
from deepeval.models.utils import (
    require_costs,
    normalize_kwargs_and_extract_aliases,
)


retry_bedrock = create_retry_decorator(PS.BEDROCK)

_ALIAS_MAP = {
    "model": ["model_id"],
    "region": ["region_name"],
    "cost_per_input_token": ["input_token_cost"],
    "cost_per_output_token": ["output_token_cost"],
}


class AmazonBedrockModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        region: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "AmazonBedrockModel",
            kwargs,
            _ALIAS_MAP,
        )

        # Backwards compatibility for renamed params
        if model is None and "model" in alias_values:
            model = alias_values["model"]
        if (
            cost_per_input_token is None
            and "cost_per_input_token" in alias_values
        ):
            cost_per_input_token = alias_values["cost_per_input_token"]
        if (
            cost_per_output_token is None
            and "cost_per_output_token" in alias_values
        ):
            cost_per_output_token = alias_values["cost_per_output_token"]

        # Secrets: prefer explicit args -> settings -> then AWS default chain
        if aws_access_key_id is not None:
            self.aws_access_key_id: Optional[SecretStr] = SecretStr(
                aws_access_key_id
            )
        else:
            self.aws_access_key_id = settings.AWS_ACCESS_KEY_ID

        if aws_secret_access_key is not None:
            self.aws_secret_access_key: Optional[SecretStr] = SecretStr(
                aws_secret_access_key
            )
        else:
            self.aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY

        # Dependencies: aiobotocore & botocore
        aiobotocore_session = require_dependency(
            "aiobotocore.session",
            provider_label="AmazonBedrockModel",
            install_hint="Install it with `pip install aiobotocore`.",
        )
        self.botocore_module = require_dependency(
            "botocore",
            provider_label="AmazonBedrockModel",
            install_hint="Install it with `pip install botocore`.",
        )
        self._session = aiobotocore_session.get_session()
        self._exit_stack = AsyncExitStack()

        # Defaults from settings
        model = model or settings.AWS_BEDROCK_MODEL_NAME
        region = region or settings.AWS_BEDROCK_REGION

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.AWS_BEDROCK_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.AWS_BEDROCK_COST_PER_OUTPUT_TOKEN
        )

        # Required params
        model = require_param(
            model,
            provider_label="AmazonBedrockModel",
            env_var_name="AWS_BEDROCK_MODEL_NAME",
            param_hint="model",
        )
        region = require_param(
            region,
            provider_label="AmazonBedrockModel",
            env_var_name="AWS_BEDROCK_REGION",
            param_hint="region",
        )

        self.model_data = BEDROCK_MODELS_DATA.get(model)
        cost_per_input_token, cost_per_output_token = require_costs(
            self.model_data,
            model,
            "AWS_BEDROCK_COST_PER_INPUT_TOKEN",
            "AWS_BEDROCK_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token,
            cost_per_output_token,
        )

        # Final attributes
        self.region = region
        self.cost_per_input_token = float(cost_per_input_token or 0.0)
        self.cost_per_output_token = float(cost_per_output_token or 0.0)

        self.kwargs = normalized_kwargs
        self.generation_kwargs = generation_kwargs or {}
        self._client = None
        self._sdk_retry_mode: Optional[bool] = None

        super().__init__(model)

    ###############################################
    # Generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        return safe_asyncio_run(self.a_generate(prompt, schema))

    @retry_bedrock
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            payload = self.generate_payload(prompt)
        else:
            payload = self.get_converse_request_body(prompt)

        client = await self._ensure_client()
        response = await client.converse(
            modelId=self.get_model_name(),
            messages=payload["messages"],
            inferenceConfig=payload["inferenceConfig"],
        )

        message = self._extract_text_from_converse_response(response)

        cost = self.calculate_cost(
            response["usage"]["inputTokens"],
            response["usage"]["outputTokens"],
        )
        if schema is None:
            return message, cost
        else:
            json_output = trim_and_load_json(message)
            return schema.model_validate(json_output), cost

    def generate_payload(
        self, multimodal_input: Optional[List[Union[str, MLLMImage]]] = None
    ):
        multimodal_input = [] if multimodal_input is None else multimodal_input
        content = []
        for element in multimodal_input:
            if isinstance(element, str):
                content.append({"text": element})
            elif isinstance(element, MLLMImage):
                # Bedrock doesn't support external URLs - must convert everything to bytes
                element.ensure_images_loaded()

                image_format = (
                    (element.mimeType or "image/jpeg").split("/")[-1].upper()
                )
                image_format = "JPEG" if image_format == "JPG" else image_format

                try:
                    image_raw_bytes = base64.b64decode(element.dataBase64)
                except Exception:
                    raise DeepEvalError(
                        f"Invalid base64 data in MLLMImage: {element._id}"
                    )

                content.append(
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_raw_bytes},
                        }
                    }
                )

        return {
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {
                **self.generation_kwargs,
            },
        }

    #########################
    # Capabilities          #
    #########################

    def supports_log_probs(self) -> Union[bool, None]:
        return self.model_data.supports_log_probs

    def supports_temperature(self) -> Union[bool, None]:
        return self.model_data.supports_temperature

    def supports_multimodal(self) -> Union[bool, None]:
        return self.model_data.supports_multimodal

    def supports_structured_outputs(self) -> Union[bool, None]:
        return self.model_data.supports_structured_outputs

    def supports_json_mode(self) -> Union[bool, None]:
        return self.model_data.supports_json

    ###############################################
    # Client management
    ###############################################

    async def _ensure_client(self):

        use_sdk = sdk_retries_for(PS.BEDROCK)

        # only rebuild if client is missing or the sdk retry mode changes
        if self._client is None or self._sdk_retry_mode != use_sdk:

            # create retry config for botocore
            retries_config = {"max_attempts": (5 if use_sdk else 1)}
            if use_sdk:
                retries_config["mode"] = "adaptive"

            Config = self.botocore_module.config.Config
            config = Config(retries=retries_config)

            client_kwargs = {
                "region_name": self.region,
                "config": config,
                **self.kwargs,
            }

            if self.aws_access_key_id is not None:
                client_kwargs["aws_access_key_id"] = (
                    self.aws_access_key_id.get_secret_value()
                )
            if self.aws_secret_access_key is not None:
                client_kwargs["aws_secret_access_key"] = (
                    self.aws_secret_access_key.get_secret_value()
                )

            cm = self._session.create_client("bedrock-runtime", **client_kwargs)

            self._client = await self._exit_stack.enter_async_context(cm)
            self._sdk_retry_mode = use_sdk

        return self._client

    async def close(self):
        await self._exit_stack.aclose()
        self._client = None

    ###############################################
    # Helpers
    ###############################################

    @staticmethod
    def _extract_text_from_converse_response(response: dict) -> str:
        try:
            content = response["output"]["message"]["content"]
        except Exception as e:
            raise DeepEvalError(
                "Missing output.message.content in Bedrock response"
            ) from e

        # Collect any text blocks (ignore reasoning/tool blocks)
        text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                v = block.get("text")
                if isinstance(v, str) and v.strip():
                    text_parts.append(v)

        if text_parts:
            # join in case there are multiple text blocks
            return "\n".join(text_parts)

        # No text blocks present; raise an actionable error
        keys = []
        for b in content:
            if isinstance(b, dict):
                keys.append(list(b.keys()))
            else:
                keys.append(type(b).__name__)

        stop_reason = (
            response.get("stopReason")
            or response.get("output", {}).get("stopReason")
            or response.get("output", {}).get("message", {}).get("stopReason")
        )

        raise DeepEvalError(
            f"Bedrock response contained no text content blocks. "
            f"content keys={keys}, stopReason={stop_reason}"
        )

    def get_converse_request_body(self, prompt: str) -> dict:

        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                **self.generation_kwargs,
            },
        }

    def calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        if self.model_data.input_price and self.model_data.output_price:
            input_cost = input_tokens * self.model_data.input_price
            output_cost = output_tokens * self.model_data.output_price
            return input_cost + output_cost
        return None

    def load_model(self):
        pass

    def get_model_name(self) -> str:
        return self.name
