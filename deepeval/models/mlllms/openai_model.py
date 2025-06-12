from typing import Optional, Tuple, List, Union
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel
from io import BytesIO
import logging
import openai
import base64
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    RetryCallState,
)

from deepeval.models import DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage
from deepeval.models.utils import parse_model_name

retryable_exceptions = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.LengthFinishReasonError,
)


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"OpenAI Error: {exception} Retrying: {retry_state.attempt_number} time(s)..."
    )


model_pricing = {
    "gpt-4o-mini": {"input": 0.150 / 1e6, "output": 0.600 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4-turbo": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-turbo-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-0125-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-1106-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4": {"input": 30.00 / 1e6, "output": 60.00 / 1e6},
    "gpt-4-32k": {"input": 60.00 / 1e6, "output": 120.00 / 1e6},
    "gpt-3.5-turbo-1106": {"input": 1.00 / 1e6, "output": 2.00 / 1e6},
    "gpt-3.5-turbo": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
    "gpt-3.5-turbo-16k": {"input": 3.00 / 1e6, "output": 4.00 / 1e6},
    "gpt-3.5-turbo-0125": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
    "gpt-3.5-turbo-instruct": {"input": 1.50 / 1e6, "output": 2.00 / 1e6},
    "o1": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o1-preview": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o1-2024-12-17": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o3-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    "o3-mini-2025-01-31": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    "gpt-4.5-preview-2025-02-27": {
        "input": 75.00 / 1e6,
        "output": 150.00 / 1e6,
    },
}

valid_multimodal_gpt_models = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
]

default_multimodal_gpt_model = "gpt-4o"


class MultimodalOpenAIModel(DeepEvalBaseMLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = parse_model_name(model)
            if model_name not in valid_multimodal_gpt_models:
                raise ValueError(
                    f"Invalid model. Available Multimodal GPT models: {', '.join(model for model in valid_multimodal_gpt_models)}"
                )
        elif model is None:
            model_name = default_multimodal_gpt_model

        self._openai_api_key = _openai_api_key
        self.args = args
        self.kwargs = kwargs

        super().__init__(model_name, *args, **kwargs)

    ###############################################
    # Generate functions
    ###############################################

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> Tuple[str, float]:
        client = OpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)
        messages = [{"role": "user", "content": prompt}]
        response = client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=schema,
        )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_cost = self.calculate_cost(input_tokens, output_tokens)
        generated_text = response.choices[0].message.parsed
        return generated_text, total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> Tuple[str, float]:
        client = AsyncOpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)
        messages = [{"role": "user", "content": prompt}]
        response = await client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=schema,
        )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_cost = self.calculate_cost(input_tokens, output_tokens)
        generated_text = response.choices[0].message.parsed
        return generated_text, total_cost

    ###############################################
    # Other generate functions
    ###############################################

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_raw_response(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        top_logprobs: int = 5,
    ) -> Tuple[ParsedChatCompletion, float]:
        client = OpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        return completion, cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate_raw_response(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        top_logprobs: int = 5,
    ) -> Tuple[ParsedChatCompletion, float]:
        client = AsyncOpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)
        messages = [{"role": "user", "content": prompt}]
        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)
        return completion, cost

    ###############################################
    # Utilities
    ###############################################

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

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(
            self.model_name, model_pricing["gpt-4o"]
        )  # Default to 'gpt-4o' if model not found
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    def encode_pil_image(self, pil_image):
        image_buffer = BytesIO()
        if pil_image.mode in ("RGBA", "LA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return base64_encoded_image

    def get_model_name(self):
        return self.model_name
