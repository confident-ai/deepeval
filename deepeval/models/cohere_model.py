from typing import Optional, Union

from cohere import Client
from deepeval.models.base_model import DeepEvalBaseLLM


class CohereModel(DeepEvalBaseLLM):
    def __init__(
            self,
            _openai_api_key: Optional[str] = None,
            _model_name: Optional[str] = "command-r",
    ):
        self.cohere_api_key = _openai_api_key
        self.model_name = _model_name
        self.max_tokens = 1024
        self.temperature = 0.7
        super().__init__()

    def load_model(self):
        return Client(self.cohere_api_key)

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.chat(message=prompt,
                               max_tokens=self.max_tokens,
                               temperature=self.temperature).text

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.chat(message=prompt,
                                    max_tokens=self.max_tokens,
                                    temperature=self.temperature)
        return res.text

    def get_model_name(self):
        return self.model_name
