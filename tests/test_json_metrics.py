import asyncio
from pydantic import BaseModel

import logging
import openai
import instructor
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from typing import Optional, Tuple
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseLLM


def log_retry_error(retry_state):
    logging.error(
        f"OpenAI rate limit exceeded. Retrying: {retry_state.attempt_number} time(s)..."
    )


valid_gpt_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0125",
]

default_gpt_model = "gpt-4o"

#########################################################
##### custom model with pydantic_model argument
#########################################################


class CustomGPTEnforced(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_gpt_models:
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        self.is_azure_model: bool
        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        if self.should_use_azure_openai():
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_API_KEY
            )

            openai_api_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.OPENAI_API_VERSION
            )
            azure_deployment = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_DEPLOYMENT_NAME
            )
            azure_endpoint = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_ENDPOINT
            )

            model_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_MODEL_VERSION
            )

            if model_version is None:
                model_version = ""

            return AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_version=model_version,
                *self.args,
                **self.kwargs,
            )

        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self._openai_api_key,
            *self.args,
            **self.kwargs,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        client = instructor.from_openai(OpenAI())
        return client.chat.completions.create(
            model=self.model_name,
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        client = instructor.from_openai(AsyncOpenAI())
        response = await client.chat.completions.create(
            model=self.model_name,
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
        )
        return response

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.model_name:
            return self.model_name


#########################################################
##### custom model with no pydantic_model argument
#########################################################


class CustomGPT(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = None
        if isinstance(model, str):
            model_name = model
            if model_name not in valid_gpt_models:
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        self.is_azure_model: bool
        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        if self.should_use_azure_openai():
            openai_api_key = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_API_KEY
            )

            openai_api_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.OPENAI_API_VERSION
            )
            azure_deployment = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_DEPLOYMENT_NAME
            )
            azure_endpoint = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_OPENAI_ENDPOINT
            )

            model_version = KEY_FILE_HANDLER.fetch_data(
                KeyValues.AZURE_MODEL_VERSION
            )

            if model_version is None:
                model_version = ""

            return AzureChatOpenAI(
                openai_api_version=openai_api_version,
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                openai_api_key=openai_api_key,
                model_version=model_version,
                *self.args,
                **self.kwargs,
            )

        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self._openai_api_key,
            *self.args,
            **self.kwargs,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(
        self,
        prompt: str,
    ) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res.content

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(
        self,
        prompt: str,
    ) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
            return res.content

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.model_name:
            return self.model_name


#########################################################
##### Define Metrics
#########################################################

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    SummarizationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is this",
    actual_output="this is a latte",
    expected_output="this is a mocha",
    retrieval_context=["I love coffee"],
    context=["I love coffee"],
)

answer_relevancy = AnswerRelevancyMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
answer_relevancy_non_confine = AnswerRelevancyMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
answer_relevancy_confine = AnswerRelevancyMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

bias = BiasMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
bias_non_confine = BiasMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
bias_confine = BiasMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

contextual_precision = ContextualPrecisionMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
contextual_precision_non_confine = ContextualPrecisionMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
contextual_precision_confine = ContextualPrecisionMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

contextual_recall = ContextualRecallMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
contextual_recall_non_confine = ContextualRecallMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
contextual_recall_confine = ContextualRecallMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

contextual_relevancy = ContextualRelevancyMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
contextual_relevancy_non_confine = ContextualRelevancyMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
contextual_relevancy_confine = ContextualRelevancyMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

faithfulness = FaithfulnessMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
faithfulness_non_confine = FaithfulnessMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
faithfulness_confine = FaithfulnessMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

hallucination = HallucinationMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
hallucination_non_confine = HallucinationMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
hallucination_confine = HallucinationMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

summarization = SummarizationMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
summarization_non_confine = SummarizationMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
summarization_confine = SummarizationMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

toxicity = ToxicityMetric(
    verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125"
)
toxicity_non_confine = ToxicityMetric(
    verbose_mode=True, threshold=0.5, model=CustomGPT("gpt-3.5-turbo-0125")
)
toxicity_confine = ToxicityMetric(
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

g_eval = GEval(
    name="coherence",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Response should be concise",
        "Reponse should read easily",
    ],
    verbose_mode=True,
    threshold=0.5,
    model="gpt-3.5-turbo-0125",
)
g_eval_non_confine = GEval(
    name="coherence",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Response should be concise",
        "Reponse should read easily",
    ],
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPT("gpt-3.5-turbo-0125"),
)
g_eval_confine = GEval(
    name="coherence",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Response should be concise",
        "Reponse should read easily",
    ],
    verbose_mode=True,
    threshold=0.5,
    model=CustomGPTEnforced("gpt-3.5-turbo-0125"),
)

#########################################################
##### measure Metrics
#########################################################

# answer_relevancy.measure(test_case)
# answer_relevancy_non_confine.measure(test_case)
# answer_relevancy_confine.measure(test_case)

# bias.measure(test_case)
# bias_non_confine.measure(test_case)
# bias_confine.measure(test_case)

# contextual_precision.measure(test_case)
# contextual_precision_confine.measure(test_case)
# contextual_precision_non_confine.measure(test_case)

# contextual_recall.measure(test_case)
# contextual_recall_confine.measure(test_case)
# contextual_recall_non_confine.measure(test_case)

# contextual_relevancy.measure(test_case)
# contextual_relevancy_confine.measure(test_case)
# contextual_relevancy_non_confine.measure(test_case)

# faithfulness.measure(test_case)
# faithfulness_confine.measure(test_case)
# faithfulness_non_confine.measure(test_case)

# hallucination.measure(test_case)
# hallucination_confine.measure(test_case)
# hallucination_non_confine.measure(test_case)

# summarization.measure(test_case)
# summarization_confine.measure(test_case)
# summarization_non_confine.measure(test_case)

# toxicity.measure(test_case)
# toxicity_confine.measure(test_case)
# toxicity_non_confine.measure(test_case)

# g_eval.measure(test_case)
# g_eval_confine.measure(test_case)
# g_eval_non_confine.measure(test_case)


#########################################################
##### a_measure metrics
#########################################################

import asyncio


async def test_answer_relevancy():
    result = await answer_relevancy.a_measure(test_case)
    print("Answer Relevancy:", result)


async def test_answer_relevancy_non_confine():
    result = await answer_relevancy_non_confine.a_measure(test_case)
    print("Answer Relevancy Non Confine:", result)


async def test_answer_relevancy_confine():
    result = await answer_relevancy_confine.a_measure(test_case)
    print("Answer Relevancy Confine:", result)


async def test_bias():
    result = await bias.a_measure(test_case)
    print("Bias:", result)


async def test_bias_non_confine():
    result = await bias_non_confine.a_measure(test_case)
    print("Bias Non Confine:", result)


async def test_bias_confine():
    result = await bias_confine.a_measure(test_case)
    print("Bias Confine:", result)


async def test_contextual_recall():
    result = await contextual_recall.a_measure(test_case)
    print("Contextual Recall:", result)


async def test_contextual_recall_non_confine():
    result = await contextual_recall_non_confine.a_measure(test_case)
    print("Contextual Recall Non Confine:", result)


async def test_contextual_recall_confine():
    result = await contextual_recall_confine.a_measure(test_case)
    print("Contextual Recall Confine:", result)


async def test_contextual_relevancy():
    result = await contextual_relevancy.a_measure(test_case)
    print("Contextual Relevancy:", result)


async def test_contextual_relevancy_non_confine():
    result = await contextual_relevancy_non_confine.a_measure(test_case)
    print("Contextual Relevancy Non Confine:", result)


async def test_contextual_relevancy_confine():
    result = await contextual_relevancy_confine.a_measure(test_case)
    print("Contextual Relevancy Confine:", result)


async def test_faithfulness():
    result = await faithfulness.a_measure(test_case)
    print("Faithfulness:", result)


async def test_faithfulness_non_confine():
    result = await faithfulness_non_confine.a_measure(test_case)
    print("Faithfulness Non Confine:", result)


async def test_faithfulness_confine():
    result = await faithfulness_confine.a_measure(test_case)
    print("Faithfulness Confine:", result)


async def test_hallucination():
    result = await hallucination.a_measure(test_case)
    print("Hallucination:", result)


async def test_hallucination_non_confine():
    result = await hallucination_non_confine.a_measure(test_case)
    print("Hallucination Non Confine:", result)


async def test_hallucination_confine():
    result = await hallucination_confine.a_measure(test_case)
    print("Hallucination Confine:", result)


async def test_summarization():
    result = await summarization.a_measure(test_case)
    print("Summarization:", result)


async def test_summarization_non_confine():
    result = await summarization_non_confine.a_measure(test_case)
    print("Summarization Non Confine:", result)


async def test_summarization_confine():
    result = await summarization_confine.a_measure(test_case)
    print("Summarization Confine:", result)


async def test_toxicity():
    result = await toxicity.a_measure(test_case)
    print("Toxicity:", result)


async def test_toxicity_non_confine():
    result = await toxicity_non_confine.a_measure(test_case)
    print("Toxicity Non Confine:", result)


async def test_toxicity_confine():
    result = await toxicity_confine.a_measure(test_case)
    print("Toxicity Confine:", result)


async def test_g_eval():
    result = await g_eval.a_measure(test_case)
    print("G Eval:", result)


async def test_g_eval_non_confine():
    result = await g_eval_non_confine.a_measure(test_case)
    print("G Eval Non Confine:", result)


async def test_g_eval_confine():
    result = await g_eval_confine.a_measure(test_case)
    print("G Eval Confine:", result)


async def main():
    await asyncio.gather(
        test_answer_relevancy(),
        test_answer_relevancy_non_confine(),
        test_answer_relevancy_confine(),
        test_bias(),
        test_bias_non_confine(),
        test_bias_confine(),
        test_contextual_recall(),
        test_contextual_recall_non_confine(),
        test_contextual_recall_confine(),
        test_contextual_relevancy(),
        test_contextual_relevancy_non_confine(),
        test_contextual_relevancy_confine(),
        test_faithfulness(),
        test_faithfulness_non_confine(),
        test_faithfulness_confine(),
        test_hallucination(),
        test_hallucination_non_confine(),
        test_hallucination_confine(),
        test_summarization(),
        test_summarization_non_confine(),
        test_summarization_confine(),
        test_toxicity(),
        test_toxicity_non_confine(),
        test_toxicity_confine(),
        test_g_eval(),
        test_g_eval_non_confine(),
        test_g_eval_confine(),
    )


# Run the main function
asyncio.run(main())
