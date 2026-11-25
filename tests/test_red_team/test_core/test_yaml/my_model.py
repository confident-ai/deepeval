from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM


class CustomModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = "CustomModel"

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return prompt

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "CustomModel"
