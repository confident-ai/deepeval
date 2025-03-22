from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import json

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


class JSONCustomModel(DeepEvalBaseLLM):
    def __init__(self):
        model_name = "gpt-4o"
        super().__init__(model_name)

    def generate(self, prompt: str, schema: BaseModel):
        client = OpenAI()
        completion = client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
        )
        structured_output: BaseModel = completion.choices[0].message.parsed
        return structured_output

    async def a_generate(self, prompt: str, schema: BaseModel):
        client = AsyncOpenAI()
        completion = await client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format=schema,
        )
        structured_output: BaseModel = completion.choices[0].message.parsed
        return structured_output

    def load_model(self):
        return self.get_model_name()

    def get_model_name(self):
        return self.model_name


class CustomModel(DeepEvalBaseLLM):
    def __init__(self):
        model_name = "gpt-3.5-turbo"
        super().__init__(model_name)

    def generate(self, prompt: str):
        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        output = completion.choices[0].message.content
        return output

    async def a_generate(self, prompt: str):
        client = AsyncOpenAI()
        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        output = completion.choices[0].message.content
        return output

    def load_model(self):
        return self.get_model_name()

    def get_model_name(self):
        return self.model_name
