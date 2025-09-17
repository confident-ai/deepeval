from typing import Optional, Tuple, List, Union
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel
from io import BytesIO
import base64

from deepeval.models.llms.openai_model import (
    model_pricing,
    structured_outputs_models,
)
from deepeval.models import DeepEvalBaseMLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.test_case import MLLMImage
from deepeval.models.utils import parse_model_name
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.constants import ProviderSlug as PS


retry_openai = create_retry_decorator(PS.OPENAI)

valid_multimodal_gpt_models = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "gpt-4.5-preview-2025-02-27",
    "o4-mini",
]

default_multimodal_gpt_model = "gpt-4.1"

unsupported_log_probs_multimodal_gpt_models = [
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "gpt-4.5-preview-2025-02-27",
    "o4-mini",
]


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

    @retry_openai
    def generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> Tuple[str, float]:
        client = OpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)

        if schema:
            if self.model_name in structured_outputs_models:
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

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
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
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> Tuple[str, float]:
        client = AsyncOpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)

        if schema:
            if self.model_name in structured_outputs_models:
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

        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
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
        multimodal_input: List[Union[str, MLLMImage]],
        top_logprobs: int = 5,
    ) -> Tuple[ParsedChatCompletion, float]:
        client = self._client()
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

    @retry_openai
    async def a_generate_raw_response(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        top_logprobs: int = 5,
    ) -> Tuple[ParsedChatCompletion, float]:
        client = self._client(async_mode=True)
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
            self.model_name, model_pricing["gpt-4.1"]
        )  # Default to 'gpt-4.1' if model not found
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

    def _client(self, async_mode: bool = False):
        kw = {"api_key": self._openai_api_key}
        if not sdk_retries_for(PS.OPENAI):
            kw["max_retries"] = 0
        Client = AsyncOpenAI if async_mode else OpenAI
        return Client(**kw)

    def get_model_name(self):
        return self.model_name
