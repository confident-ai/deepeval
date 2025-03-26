import pytest
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.llms import (
    GPTModel,
    AzureOpenAIModel,
    OllamaModel,
    LocalModel,
)
from pydantic import BaseModel

# Sample input
input_text = """What is the best city in the world? 
Please generate a json with two keys: city, and country, both strings. For example: 
{
    "city": San Francisco,
    "country": USA
}
"""


# Schema for structured output
class City(BaseModel):
    city: str
    country: str


@pytest.mark.parametrize(
    "model_class", [GPTModel, AzureOpenAIModel, OllamaModel, LocalModel]
)
def test_generate_without_schema_returns_string(model_class):
    model: DeepEvalBaseLLM = model_class()
    output, _ = model.generate(input_text)
    assert isinstance(
        output, str
    ), f"{model_class.__name__} should return a string when no schema is provided"


@pytest.mark.parametrize(
    "model_class", [GPTModel, AzureOpenAIModel, OllamaModel, LocalModel]
)
def test_generate_with_schema_returns_city_object(model_class):
    model: DeepEvalBaseLLM = model_class()
    output, _ = model.generate(input_text, City)
    assert isinstance(
        output, City
    ), f"{model_class.__name__} should return a City object when schema is provided"
    assert isinstance(output.city, str)
    assert isinstance(output.country, str)
