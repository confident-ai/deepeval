from langchain_community.callbacks import get_openai_callback
from typing import Optional, Tuple, Union, Dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


class LocalModel(DeepEvalBaseLLM):
    def __init__(
        self,
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
        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        if schema:
            chat_model = self.load_model()
            with get_openai_callback() as cb:
                res = chat_model.invoke(prompt)
                json_output = trim_and_load_json(res.content)
                return schema.model_validate(json_output), cb.total_cost
        else:
            chat_model = self.load_model()
            with get_openai_callback() as cb:
                res = chat_model.invoke(prompt)
                return res.content, cb.total_cost

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        if schema:
            chat_model = self.load_model()
            with get_openai_callback() as cb:
                res = await chat_model.ainvoke(prompt)
                json_output = trim_and_load_json(res.content)
                return schema.model_validate(json_output), cb.total_cost
        else:
            chat_model = self.load_model()
            with get_openai_callback() as cb:
                res = await chat_model.ainvoke(prompt)
                return res.content, cb.total_cost

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        return f"{model_name} (Local Model)"

    def load_model(self):
        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.local_model_api_key,
            base_url=self.base_url,
            temperature=0,
            *self.args,
            **self.kwargs,
        )
