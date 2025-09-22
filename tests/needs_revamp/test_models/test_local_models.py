from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
import pytest
import os

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models.llms import (
    OllamaModel,
    LocalModel,
)
from deepeval.cli.main import set_local_model_env, set_ollama_model_env

# Write to Key File
load_dotenv()
for key, value in os.environ.items():
    print(key, value)


# Ollama
def set_ollama():
    set_ollama_model_env(
        model_name=os.environ["OLLAMA_MODEL_NAME"],
        base_url="http://localhost:11434",
    )


# LMStudio
def set_lm_studio():
    set_local_model_env(
        model_name=os.environ["LM_STUDIO_MODEL_NAME"],
        base_url="http://localhost:1234/v1/",
        api_key=os.environ["LM_STUDIO_API_KEY"],
        model_format=None,
    )


# vLLM
def set_vllm():
    set_local_model_env(
        model_name=os.environ["VLLM_MODEL_NAME"],
        base_url="http://localhost:8000/v1/",
        api_key=os.environ["VLLM_API_KEY"],
        model_format=None,
    )


# Clear
def clear():
    for key in KeyValues:
        if key == "api_key" or key == "confident_region":
            continue
        KEY_FILE_HANDLER.remove_key(key)


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
    "model_class,setup_func",
    [
        (OllamaModel, set_ollama),
        (LocalModel, set_lm_studio),
        (LocalModel, set_vllm),
    ],
)
def test_generate_without_schema_returns_string(model_class, setup_func):
    try:
        if setup_func:
            setup_func()
        model: DeepEvalBaseLLM = model_class()
        output, _ = model.generate(input_text)
        assert isinstance(
            output, str
        ), f"{model_class.__name__} should return a string when no schema is provided"
    finally:
        clear()


@pytest.mark.parametrize(
    "model_class,setup_func",
    [
        (OllamaModel, set_ollama),
        (LocalModel, set_lm_studio),
        (LocalModel, set_vllm),
    ],
)
def test_a_generate_without_schema_returns_string(model_class, setup_func):
    try:
        if setup_func:
            setup_func()

        model: DeepEvalBaseLLM = model_class()
        output, _ = asyncio.run(model.a_generate(input_text))
        assert isinstance(
            output, str
        ), f"{model_class.__name__} should return a string when no schema is provided"
    finally:
        clear()


@pytest.mark.parametrize(
    "model_class,setup_func",
    [
        (OllamaModel, set_ollama),
        (LocalModel, set_lm_studio),
        (LocalModel, set_vllm),
    ],
)
def test_generate_with_schema_returns_city_object(model_class, setup_func):
    try:
        if setup_func:
            setup_func()

        model: DeepEvalBaseLLM = model_class()
        output, _ = model.generate(input_text, City)
        assert isinstance(
            output, City
        ), f"{model_class.__name__} should return a City object when schema is provided"
        assert isinstance(output.city, str)
        assert isinstance(output.country, str)
    finally:
        clear()


@pytest.mark.parametrize(
    "model_class,setup_func",
    [
        (OllamaModel, set_ollama),
        (LocalModel, set_lm_studio),
        (LocalModel, set_vllm),
    ],
)
def test_a_generate_with_schema_returns_city_object(model_class, setup_func):
    try:
        if setup_func:
            setup_func()

        model: DeepEvalBaseLLM = model_class()
        output, _ = asyncio.run(model.a_generate(input_text, City))
        assert isinstance(
            output, City
        ), f"{model_class.__name__} should return a City object when schema is provided"
        assert isinstance(output.city, str)
        assert isinstance(output.country, str)
    finally:
        clear()


@pytest.mark.parametrize(
    "model_class,setup_func",
    [
        (OllamaModel, set_ollama),
        (LocalModel, set_lm_studio),
        (LocalModel, set_vllm),
    ],
)
def test_answer_relevancy_measure_sync(model_class, setup_func):
    try:
        if setup_func:
            setup_func()

        model = model_class()
        question = "What is the capital of Germany?"
        answer = "The capital of Germany is Berlin. It's a historic city with rich culture."
        test_case = LLMTestCase(input=question, actual_output=answer)
        metric = AnswerRelevancyMetric(model=model, async_mode=False)
        score = metric.measure(test_case, _show_indicator=False)
        assert isinstance(score, float)
        assert 0 <= score <= 1
    finally:
        clear()


@pytest.mark.parametrize(
    "model_class,setup_func",
    [
        (OllamaModel, set_ollama),
        (LocalModel, set_lm_studio),
        (LocalModel, set_vllm),
    ],
)
def test_answer_relevancy_measure_async(model_class, setup_func):
    try:
        if setup_func:
            setup_func()

        model = model_class()
        question = "What is the capital of Germany?"
        answer = "The capital of Germany is Berlin. It's a historic city with rich culture."
        test_case = LLMTestCase(input=question, actual_output=answer)
        metric = AnswerRelevancyMetric(model=model, async_mode=True)
        score = metric.measure(test_case, _show_indicator=False)
        assert isinstance(score, float)
    finally:
        clear()


@pytest.mark.parametrize(
    "model_class,setup_func",
    [
        (OllamaModel, set_ollama),
        (LocalModel, set_lm_studio),
        (LocalModel, set_vllm),
    ],
)
def test_evaluate_function(model_class, setup_func):
    try:
        if setup_func:
            setup_func()

        model = model_class()
        question1 = "What is the capital of Japan?"
        relevant_answer = "The capital of Japan is Tokyo. It is the largest metropolitan area in the world."
        question2 = "What is the capital of Italy?"
        irrelevant_answer = "Pizza was invented in Naples, Italy. It's a popular dish worldwide."
        test_case1 = LLMTestCase(input=question1, actual_output=relevant_answer)
        test_case2 = LLMTestCase(
            input=question2, actual_output=irrelevant_answer
        )
        evaluate(
            test_cases=[test_case1],
            metrics=[AnswerRelevancyMetric(model=model)],
        )
        assert True
    finally:
        clear()
