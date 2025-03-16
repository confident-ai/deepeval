from langchain_community.callbacks import get_openai_callback
from typing import Optional, Tuple, List, Union, Dict
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import json
import re


from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


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

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
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
    # Utilities
    ###############################################

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
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        return f"{model_name} (Local Model)"

    def load_model(self, enforce_json: bool = False):
        format = "json" if enforce_json else None
        return CustomChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.local_model_api_key,
            base_url=self.base_url,
            format=format,
            temperature=0,
            *self.args,
            **self.kwargs,
        )
