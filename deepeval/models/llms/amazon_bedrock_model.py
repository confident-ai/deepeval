from typing import Optional, Tuple, Union, Dict, List
from contextlib import AsyncExitStack
from pydantic import BaseModel
import asyncio

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json

# check aiobotocore availability
try:
    from aiobotocore.session import get_session
    from botocore.config import Config

    aiobotocore_available = True
except ImportError:
    aiobotocore_available = False


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
        self._config = Config(retries={"max_attempts": 5, "mode": "adaptive"})
        self._exit_stack = AsyncExitStack()
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        self._client = None

    ###############################################
    # Generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        return asyncio.run(self.a_generate(prompt, schema))

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

    def chat_generate(
        self, messages: List[Dict[str, str]], schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        return asyncio.run(self.a_chat_generate(messages, schema))

    async def a_chat_generate(
        self, messages: List[Dict[str, str]], schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        payload = self.get_converse_request_body_from_messages(messages)
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
        if self._client is None:
            cm = self._session.create_client(
                "bedrock-runtime",
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                config=self._config,
                **self.kwargs,
            )
            self._client = await self._exit_stack.enter_async_context(cm)
        return self._client

    async def close(self):
        await self._exit_stack.aclose()
        self._client = None

    ###############################################
    # Helpers
    ###############################################

    def get_converse_request_body(self, prompt: str) -> dict:
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": self.temperature,
                "topP": self.generation_kwargs.get("top_p", 0),
                "maxTokens": self.generation_kwargs.get("max_tokens", 1000),
                **self.generation_kwargs,
            },
        }

    def get_converse_request_body_from_messages(self, messages: List[Dict[str, str]]) -> dict:
        # Convert message format to Bedrock format
        bedrock_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map roles to Bedrock format
            if role == "assistant":
                role = "assistant"
            elif role == "system":
                # Bedrock handles system messages differently, we'll prepend to user content
                role = "user"
                content = f"[System]: {content}"
            else:
                role = "user"
            
            bedrock_messages.append({"role": role, "content": [{"text": content}]})
        
        return {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "temperature": self.temperature,
                "topP": self.generation_kwargs.get("top_p", 0),
                "maxTokens": self.generation_kwargs.get("max_tokens", 1000),
                **self.generation_kwargs,
            },
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
