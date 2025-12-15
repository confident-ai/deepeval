import base64
from typing import Optional, Tuple, Union, Dict, List
from contextlib import AsyncExitStack
from pydantic import BaseModel
from io import BytesIO
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json, safe_asyncio_run
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
        self.input_token_cost = input_token_cost
        self.output_token_cost = output_token_cost

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
        return safe_asyncio_run(self.a_generate(prompt, schema))

    @retry_bedrock
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            payload = self.generate_payload(prompt)
        else:
            payload = self.get_converse_request_body(prompt)

        try:
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
        finally:
            await self.close()

    def generate_payload(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ):
        content = []
        for element in multimodal_input:
            if isinstance(element, str):
                content.append({"text": element})
            elif isinstance(element, MLLMImage):
                # Bedrock doesn't support external URLs - must convert everything to bytes
                element.ensure_images_loaded()
                
                image_format = (element.mimeType or "image/jpeg").split("/")[-1].upper()
                image_format = "JPEG" if image_format == "JPG" else image_format
                
                try:
                    image_raw_bytes = base64.b64decode(element.dataBase64)
                except Exception:
                    raise ValueError(f"Invalid base64 data in MLLMImage: {element._id}")
                
                content.append({
                    "image": {
                        "format": image_format,
                        "source": {"bytes": image_raw_bytes},
                    }
                })
        
        return {
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {
                **self.generation_kwargs,
            },
        }

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
            fmt = (image.mimeType or "image/jpeg").split("/")[-1].upper()
            fmt = "JPEG" if fmt == "JPG" else fmt
            img.save(buf, format=fmt)
            raw_bytes = buf.getvalue()

            return fmt, raw_bytes
        if image.url:
            import requests

            resp = requests.get(image.url)
            resp.raise_for_status()
            raw_bytes = resp.content
            mime = resp.headers.get(
                "content-type", image.mimeType or "image/jpeg"
            )
            fmt = mime.split("/")[-1]
            return fmt, raw_bytes
        raise ValueError(
            "MLLMImage must contain dataBase64, or (local=True + filename), or url."
        )

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

        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
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

    def supports_multimodal(self):
        return True

    def get_model_name(self) -> str:
        return self.model_id
