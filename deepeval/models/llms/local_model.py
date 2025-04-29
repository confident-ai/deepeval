from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


class LocalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        temperature: float = 0,
        *args,
        **kwargs,
    ):
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        self.local_model_api_key = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_MODEL_API_KEY
        )
        self.base_url = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_MODEL_BASE_URL
        )
        self.format = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_FORMAT)
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        response: ChatCompletion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=True)
        response: ChatCompletion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        return f"{model_name} (Local Model)"

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return OpenAI(
                api_key=self.local_model_api_key,
                base_url=self.base_url,
                *self.args,
                **self.kwargs,
            )
        else:
            return AsyncOpenAI(
                api_key=self.local_model_api_key,
                base_url=self.base_url,
                *self.args,
                **self.kwargs,
            )
