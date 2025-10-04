import asyncio

from typing import Optional, Tuple, Union, Dict
from contextlib import AsyncExitStack
from pydantic import BaseModel

from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.constants import ProviderSlug as PS

# check aiobotocore availability
try:
    from aiobotocore.session import get_session
    from botocore.config import Config

    aiobotocore_available = True
except ImportError:
    aiobotocore_available = False

# define retry policy
retry_bedrock = create_retry_decorator(PS.BEDROCK)


def _check_aiobotocore_available():
    if not aiobotocore_available:
        raise ImportError(
            "aiobotocore and botocore are required for this functionality. "
            "Install them via your package manager (e.g. pip install aiobotocore botocore)"
        )


class AmazonBedrockModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_id: str,
        region_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        temperature: float = 0,
        input_token_cost: float = 0,
        output_token_cost: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        _check_aiobotocore_available()
        super().__init__(model_id)

        self.model_id = model_id
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.temperature = temperature
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost

        if self.temperature < 0:
            raise ValueError("Temperature must be >= 0.")

        # prepare aiobotocore session, config, and async exit stack
        self._session = get_session()
        self._exit_stack = AsyncExitStack()
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        self._client = None
        self._sdk_retry_mode: Optional[bool] = None

    ###############################################
    # Generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        return asyncio.run(self.a_generate(prompt, schema))

    @retry_bedrock
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        payload = self.get_converse_request_body(prompt)
        client = await self._ensure_client()
        response = await client.converse(
            modelId=self.model_id,
            messages=payload["messages"],
            inferenceConfig=payload["inferenceConfig"],
        )
        message = response["output"]["message"]["content"][0]["text"]
        cost = self.calculate_cost(
            response["usage"]["inputTokens"],
            response["usage"]["outputTokens"],
        )
        if schema is None:
            return message, cost
        else:
            json_output = trim_and_load_json(message)
            return schema.model_validate(json_output), cost

    ###############################################
    # Client management
    ###############################################

    async def _ensure_client(self):
        use_sdk = sdk_retries_for(PS.BEDROCK)

        # only rebuild if client is missing or the sdk retry mode changes
        if self._client is None or self._sdk_retry_mode != use_sdk:
            # Close any previous
            if self._client is not None:
                await self._exit_stack.aclose()
                self._client = None

            # create retry config for botocore
            retries_config = {"max_attempts": (5 if use_sdk else 1)}
            if use_sdk:
                retries_config["mode"] = "adaptive"

            config = Config(retries=retries_config)

            cm = self._session.create_client(
                "bedrock-runtime",
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                config=config,
                **self.kwargs,
            )
            self._client = await self._exit_stack.enter_async_context(cm)
            self._sdk_retry_mode = use_sdk

        return self._client

    async def close(self):
        await self._exit_stack.aclose()
        self._client = None

    ###############################################
    # Helpers
    ###############################################

    def get_converse_request_body(self, prompt: str) -> dict:
        # Inline parameter translation with defaults
        param_mapping = {
            "max_tokens": "maxTokens",
            "top_p": "topP",
            "top_k": "topK",
            "stop_sequences": "stopSequences",
        }

        # Start with defaults for required parameters
        translated_kwargs = {
            "maxTokens": self.generation_kwargs.get("max_tokens", 1000),
        }

        # Add any other parameters from generation_kwargs
        for key, value in self.generation_kwargs.items():
            if key not in [
                "max_tokens",
                "top_p",
            ]:  # Skip already handled defaults
                aws_key = param_mapping.get(key, key)
                translated_kwargs[aws_key] = value

        # Handle temperature and top_p compatibility for Claude Sonnet 4.5
        # Claude Sonnet 4.5 doesn't support both temperature and top_p simultaneously
        inference_config = {"maxTokens": translated_kwargs["maxTokens"]}
        
        # Check if this is Claude Sonnet 4.5
        is_claude_sonnet_4_5 = "claude-sonnet-4-5" in self.model_id.lower()
        
        if is_claude_sonnet_4_5:
            # For Claude Sonnet 4.5, prioritize temperature over top_p
            # Only use top_p if temperature is 0 and top_p is explicitly provided and > 0
            top_p_value = (
                self.generation_kwargs.get("top_p")
                or self.generation_kwargs.get("topP")
                or 0
            )
            if self.temperature == 0 and top_p_value > 0:
                inference_config["topP"] = top_p_value
                # Don't include temperature when using top_p
            else:
                # Default to temperature (including when temperature is 0)
                inference_config["temperature"] = self.temperature
        else:
            # For other models, include both parameters as before
            inference_config["temperature"] = self.temperature
            if "topP" in translated_kwargs:
                inference_config["topP"] = translated_kwargs["topP"]
            elif self.generation_kwargs.get("top_p", 0) > 0:
                inference_config["topP"] = self.generation_kwargs.get("top_p", 0)

        # Add any other translated parameters (excluding already handled ones)
        for key, value in translated_kwargs.items():
            if key not in ["maxTokens", "topP"]:
                inference_config[key] = value

        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": inference_config,
        }

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_token_cost
            + output_tokens * self.output_token_cost
        )

    def load_model(self):
        pass

    def get_model_name(self) -> str:
        return self.model_id
