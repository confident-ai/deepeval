# TODO: add CLI for anthropic and add anthropic to this test file

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
    AzureOpenAIModel,
    GPTModel,
    GeminiModel,
    KimiModel,
    GrokModel,
    DeepSeekModel,
)
from deepeval.cli.main import (
    set_openai_env,
    set_azure_openai_env,
    set_gemini_model_env,
    set_deepseek_model_env,
    set_moonshot_model_env,
    set_grok_model_env,
)

# Write to Key File
load_dotenv()
for key, value in os.environ.items():
    print(key, value)


# OpenAI
def set_openai():
    set_openai_env(
        model=os.environ["OPENAI_MODEL_NAME"],
        cost_per_input_token=os.environ["OPENAI_COST_PER_INPUT_TOKEN"],
        cost_per_output_token=os.environ["OPENAI_COST_PER_OUTPUT_TOKEN"],
    )


# Azure
def set_azure_openai():
    set_azure_openai_env(
        azure_openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        openai_model_name=os.environ["AZURE_MODEL_NAME"],
        azure_deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
        azure_model_version=os.environ["AZURE_MODEL_VERSION"],
    )


# Gemini
def set_gemini():
    set_gemini_model_env(
        model_name=os.environ["GEMINI_MODEL_NAME"],
        google_api_key=os.environ["GOOGLE_API_KEY"],
        google_cloud_project=None,
        google_cloud_location=None,
    )


# Vertex AI
def set_vertex_ai():
    set_gemini_model_env(
        model_name=os.environ["VERTEX_AI_MODEL_NAME"],
        google_api_key=None,
        google_cloud_project=os.environ["GOOGLE_CLOUD_PROJECT"],
        google_cloud_location=os.environ["GOOGLE_CLOUD_LOCATION"],
    )


# DeepSeek
def set_deepseek():
    set_deepseek_model_env(
        model_name=os.environ["DEEPSEEK_MODEL_NAME"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=os.environ["TEMPERATURE"],
    )


# Moonshot
def set_moonshot():
    set_moonshot_model_env(
        model_name=os.environ["MOONSHOT_MODEL_NAME"],
        api_key=os.environ["MOONSHOT_API_KEY"],
        temperature=os.environ["TEMPERATURE"],
    )


# Grok
def set_grok():
    set_grok_model_env(
        model_name=os.environ["GROK_MODEL_NAME"],
        api_key=os.environ["GROK_API_KEY"],
        temperature=os.environ["TEMPERATURE"],
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
        (GPTModel, set_openai),
        (AzureOpenAIModel, set_azure_openai),
        (GeminiModel, set_gemini),
        # (GeminiModel, set_vertex_ai),
        (DeepSeekModel, set_deepseek),
        (KimiModel, set_moonshot),
        (GrokModel, set_grok),
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
        (GPTModel, set_openai),
        (AzureOpenAIModel, set_azure_openai),
        (GeminiModel, set_gemini),
        # (GeminiModel, set_vertex_ai),
        (DeepSeekModel, set_deepseek),
        (KimiModel, set_moonshot),
        (GrokModel, set_grok),
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
        (GPTModel, set_openai),
        (AzureOpenAIModel, set_azure_openai),
        (GeminiModel, set_gemini),
        # (GeminiModel, set_vertex_ai),
        (DeepSeekModel, set_deepseek),
        (KimiModel, set_moonshot),
        (GrokModel, set_grok),
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
        (GPTModel, set_openai),
        (AzureOpenAIModel, set_azure_openai),
        (GeminiModel, set_gemini),
        # (GeminiModel, set_vertex_ai),
        (DeepSeekModel, set_deepseek),
        (KimiModel, set_moonshot),
        (GrokModel, set_grok),
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
        (GPTModel, set_openai),
        (AzureOpenAIModel, set_azure_openai),
        (GeminiModel, set_gemini),
        # (GeminiModel, set_vertex_ai),
        (DeepSeekModel, set_deepseek),
        (KimiModel, set_moonshot),
        (GrokModel, set_grok),
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
        (GPTModel, set_openai),
        (AzureOpenAIModel, set_azure_openai),
        (GeminiModel, set_gemini),
        # (GeminiModel, set_vertex_ai),
        (DeepSeekModel, set_deepseek),
        (KimiModel, set_moonshot),
        (GrokModel, set_grok),
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
        (GPTModel, set_openai),
        (AzureOpenAIModel, set_azure_openai),
        (GeminiModel, set_gemini),
        # (GeminiModel, set_vertex_ai),
        (DeepSeekModel, set_deepseek),
        (KimiModel, set_moonshot),
        (GrokModel, set_grok),
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
