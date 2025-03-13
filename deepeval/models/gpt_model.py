from typing import Optional, Tuple, List, Union, Dict
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from io import BytesIO
import logging
import openai
import base64
import json
import re

from tenacity import retry, retry_if_exception_type, wait_exponential_jitter
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.outputs import ChatResult
from langchain.schema import HumanMessage
from ollama import Client, AsyncClient, ChatResponse

from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseMLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.test_case import MLLMImage


def log_retry_error(retry_state):
    logging.error(
        f"OpenAI rate limit exceeded. Retrying: {retry_state.attempt_number} time(s)..."
    )


valid_gpt_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
]

unsupported_log_probs_gpt_models = [
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
]

structured_outputs_models = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o3-mini",
    "o3-mini-2025-01-31",
]

json_mode_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-32k",
    "gpt-4-32k-0613",
]

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
}

default_gpt_model = "gpt-4o"


# Adding a custom class to enable json mode in Ollama during API calls
class CustomChatOpenAI(ChatOpenAI):
    format: str = None

    def __init__(self, format: str = None, **kwargs):
        super().__init__(**kwargs)
        self.format = format

    async def _acreate(
        self, messages: List[BaseMessage], **kwargs
    ) -> ChatResult:
        if self.format:
            kwargs["format"] = self.format
        return await super()._acreate(messages, **kwargs)


class GPTModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if (
                not self.should_use_local_model()
                and model_name not in valid_gpt_models
            ):
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        self.base_url = base_url
        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        using_openai_model = (
            not self.should_use_azure_openai()
            and not self.should_use_local_model()
        )
        if using_openai_model:
            if schema:
                client = OpenAI(api_key=self._openai_api_key)
                if self.model_name in structured_outputs_models:
                    completion = client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        response_format=schema,
                    )
                    structured_output: BaseModel = completion.choices[
                        0
                    ].message.parsed
                    cost = self.calculate_cost(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                    )
                    return structured_output, cost
                if self.model_name in json_mode_models:
                    completion = client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    json_output = self.trim_and_load_json(
                        completion.choices[0].message.content
                    )
                    cost = self.calculate_cost(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                    )
                    return schema.model_validate(json_output), cost
            else:
                chat_model = self.load_model()
                with get_openai_callback() as cb:
                    res = chat_model.invoke(prompt)
                    return res.content, cb.total_cost
        elif self.should_use_ollama_model():
            chat_model = self.load_model()
            model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
            response: ChatResponse = chat_model.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                format=schema.model_json_schema() if schema else None,
            )
            return (
                (
                    schema.model_validate_json(response.message.content)
                    if schema
                    else response.message.content
                ),
                0,
            )
        elif self.should_use_local_model() or self.should_use_azure_openai():
            if schema:
                chat_model = self.load_model(enforce_json=True)
                with get_openai_callback() as cb:
                    res = chat_model.invoke(prompt)
                    json_output = self.trim_and_load_json(res.content)
                    return schema.model_validate(json_output), cb.total_cost
            else:
                chat_model = self.load_model()
                with get_openai_callback() as cb:
                    res = chat_model.invoke(prompt)
                    return res.content, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        using_openai_model = (
            not self.should_use_azure_openai()
            and not self.should_use_local_model()
        )
        if using_openai_model:
            if schema:
                client = AsyncOpenAI(api_key=self._openai_api_key)
                if self.model_name in structured_outputs_models:
                    completion = await client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        response_format=schema,
                    )
                    structured_output: BaseModel = completion.choices[
                        0
                    ].message.parsed
                    cost = self.calculate_cost(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                    )
                    return structured_output, cost
                if self.model_name in json_mode_models:
                    completion = await client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    json_output = self.trim_and_load_json(
                        completion.choices[0].message.content
                    )
                    cost = self.calculate_cost(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                    )
                    return schema.model_validate(json_output), cost
            else:
                chat_model = self.load_model()
                with get_openai_callback() as cb:
                    res = await chat_model.ainvoke(prompt)
                    return res.content, cb.total_cost
        elif self.should_use_ollama_model():
            chat_model = self.load_model(async_mode=True)
            model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
            response: ChatResponse = await chat_model.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                format=schema.model_json_schema() if schema else None,
            )
            return (
                (
                    schema.model_validate_json(response.message.content)
                    if schema
                    else response.message.content
                ),
                0,
            )
        elif self.should_use_local_model() or self.should_use_azure_openai():
            if schema:
                chat_model = self.load_model(enforce_json=True)
                with get_openai_callback() as cb:
                    res = await chat_model.ainvoke(prompt)
                    json_output = self.trim_and_load_json(res.content)
                    return schema.model_validate(json_output), cb.total_cost
            else:
                chat_model = self.load_model()
                with get_openai_callback() as cb:
                    res = await chat_model.ainvoke(prompt)
                    return res.content, cb.total_cost

    ###############################################
    # Other generate functions
    ###############################################

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate_raw_response(
        self, prompt: str, **kwargs
    ) -> Tuple[AIMessage, float]:
        if self.should_use_azure_openai():
            raise AttributeError

        chat_model = self.load_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate_raw_response(
        self, prompt: str, **kwargs
    ) -> Tuple[AIMessage, float]:
        if self.should_use_azure_openai():
            raise AttributeError

        chat_model = self.load_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
        return res, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[AIMessage, float]:
        chat_model = self.load_model()
        og_parameters = {"n": chat_model.n, "temp": chat_model.temperature}
        chat_model.n = n
        chat_model.temperature = temperature

        generations = chat_model._generate([HumanMessage(prompt)]).generations
        chat_model.temperature = og_parameters["temp"]
        chat_model.n = og_parameters["n"]

        completions = [r.text for r in generations]
        return completions

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.model_name, model_pricing["gpt-4o"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    def trim_and_load_json(
        self,
        input_string: str,
    ) -> Dict:
        start = input_string.find("{")
        end = input_string.rfind("}") + 1
        if end == 0 and start != -1:
            input_string = input_string + "}"
            end = len(input_string)
        jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
        jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
        try:
            return json.loads(jsonStr)
        except json.JSONDecodeError:
            error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
            raise ValueError(error_str)
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "Azure OpenAI"
        elif self.should_use_ollama_model():
            model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_LOCAL_MODEL)
            return f"{model_name} (Ollama)"
        elif self.should_use_local_model():
            model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
            return "{model_name} (Local Model)"
        elif self.model_name:
            return self.model_name

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def should_use_local_model(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_LOCAL_MODEL)
        return value.lower() == "yes" if value is not None else False

    def should_use_ollama_model(self):
        base_url = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_API_KEY)
        return base_url == "ollama"

    def load_model(self, enforce_json: bool = False, async_mode: bool = False):
        if self.should_use_ollama_model():
            format = "json" if enforce_json else None
            base_url = KEY_FILE_HANDLER.fetch_data(
                KeyValues.LOCAL_MODEL_BASE_URL
            )
            if not async_mode:
                return Client(host=base_url)
            else:
                return AsyncClient(host=base_url)
        elif self.should_use_local_model():
            format = "json" if enforce_json else None
            model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.LOCAL_MODEL_API_KEY
            )
            base_url = KEY_FILE_HANDLER.fetch_data(
                KeyValues.LOCAL_MODEL_BASE_URL
            )
            format = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_FORMAT)
            return CustomChatOpenAI(
                model_name=model_name,
                openai_api_key=openai_api_key,
                base_url=base_url,
                format=format,
                temperature=0,
                *self.args,
                **self.kwargs,
            )
        elif self.should_use_azure_openai():
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_API_KEY
            )
            openai_api_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.OPENAI_API_VERSION
            )
            azure_deployment = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_DEPLOYMENT_NAME
            )
            azure_endpoint = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_ENDPOINT
            )
            model_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_MODEL_VERSION
            )
            if model_version is None:
                model_version = ""
            return AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_version=model_version,
                *self.args,
                **self.kwargs,
            )
        else:
            return ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self._openai_api_key,
                base_url=self.base_url,
                *self.args,
                **self.kwargs,
            )


###############################################
# Multimodal Model
###############################################


valid_multimodal_gpt_models = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4",
    "gpt-4-0125-preview",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-32k",
    "gpt-4-32k-0613",
]

default_multimodal_gpt_model = "gpt-4o"


class MultimodalGPTModel(DeepEvalBaseMLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_multimodal_gpt_models:
                raise ValueError(
                    f"Invalid model. Available Multimodal GPT models: {', '.join(model for model in valid_multimodal_gpt_models)}"
                )
        elif model is None:
            model_name = default_multimodal_gpt_model

        self._openai_api_key = _openai_api_key
        self.args = args
        self.kwargs = kwargs
        self.model_name = model_name

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_name: str
    ) -> float:
        pricing = model_pricing.get(
            model_name, model_pricing["gpt-4o"]
        )  # Default to 'gpt-4o' if model not found
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    def calculate_image_tokens(
        self, pil_image: "PILImage", detail: str = "auto"
    ) -> int:
        width, height = pil_image.size

        def high_detail_cost() -> int:
            if max(width, height) > 2048:
                scale_factor = 2048 / max(width, height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            scale_factor = 768 / min(width, height)
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            tiles = (width // 512) * (height // 512)
            return 85 + (170 * tiles)

        if detail == "low":
            return 85
        if detail == "high":
            return high_detail_cost()
        if width > 1024 or height > 1024:
            return high_detail_cost()
        return 85

    def encode_pil_image(self, pil_image: "PILImage"):
        image_buffer = BytesIO()
        pil_image.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        base64_encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return base64_encoded_image

    def generate_prompt(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ):

        prompt = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                prompt.append({"type": "text", "text": ele})
            elif isinstance(ele, MLLMImage):
                if ele.local == True:
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

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(
        self, multimodal_input: List[Union[str, MLLMImage]]
    ) -> Tuple[str, float]:
        client = OpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_cost = self.calculate_cost(
            input_tokens, output_tokens, self.model_name
        )
        generated_text = response.choices[0].message.content
        return generated_text, total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(
        self, multimodal_input: List[Union[str, MLLMImage]]
    ) -> Tuple[str, float]:
        client = AsyncOpenAI(api_key=self._openai_api_key)
        prompt = self.generate_prompt(multimodal_input)
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_cost = self.calculate_cost(
            input_tokens, output_tokens, self.model_name
        )
        generated_text = response.choices[0].message.content
        return generated_text, total_cost

    def get_model_name(self):
        return self.model_name
