import json

from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM


class CustomJudge(DeepEvalBaseLLM):
    def __init__(self, name, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.name = name

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        result = json.loads(prompt)

        return schema(**result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name
