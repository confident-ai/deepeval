import asyncio
from typing import List
from pydantic import BaseModel
from deepeval.models import GPTModel

import logging
import openai
import instructor
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from typing import Optional, Tuple
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseLLM


def log_retry_error(retry_state):
    logging.error(
        f"OpenAI rate limit exceeded. Retrying: {retry_state.attempt_number} time(s)..."
    )


valid_gpt_models = [
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


class GPTPydanticModel(DeepEvalBaseLLM):
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
        pydantic_model: Optional[BaseModel] = None
        ) -> Tuple[str, float]:
        if pydantic_model == None:
            chat_model = self.load_model()
            with get_openai_callback() as cb:
                res = chat_model.invoke(prompt)
                return res.content, cb.total_cost
        elif pydantic_model != None:
            client = instructor.from_openai(OpenAI())
            return client.chat.completions.create(
                model=self.model_name,
                response_model=pydantic_model,
                messages=[{"role": "user", "content": prompt}],
            )

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(self, 
        prompt: str, 
        pydantic_model: Optional[BaseModel] = None
        ) -> Tuple[str, float]:
        if pydantic_model == None:
            chat_model = self.load_model()
            with get_openai_callback() as cb:
                res = await chat_model.ainvoke(prompt)
                return res.content, cb.total_cost
        else:
            client = instructor.from_openai(AsyncOpenAI())
            response = await client.chat.completions.create(
                model=self.model_name,
                response_model=pydantic_model,
                messages=[{"role": "user", "content": prompt}],
            )            
            return response

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[AIMessage, float]:
        chat_model = self.load_model()
        og_parameters = {"n": chat_model.n, "temp": chat_model.temperature}
        chat_model.n = n
        chat_model.temperature = temperature

        generations = chat_model._generate([HumanMessage(prompt)]).generations
        chat_model.temperature = og_parameters["temp"]
        chat_model.n = og_parameters["n"]

        completions = [r.text for r in generations]
        return completions
    
    # @retry(
    #     wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
    #     retry=retry_if_exception_type(openai.RateLimitError),
    #     after=log_retry_error,
    # )
    # def generate_raw_response(
    #     self, 
    #     prompt: str, 
    #     pydantic_model: Optional[BaseModel] = None,
    #     **kwargs
    # ) -> Tuple[AIMessage, float]:
    #     if self.should_use_azure_openai():
    #         raise AttributeError

    #     chat_model = self.load_model().bind(**kwargs)
    #     with get_openai_callback() as cb:
    #         res = chat_model.invoke(prompt)
    #         return res, cb.total_cost

    # @retry(
    #     wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
    #     retry=retry_if_exception_type(openai.RateLimitError),
    #     after=log_retry_error,
    # )
    # async def a_generate_raw_response(
    #     self, 
    #     prompt: str, 
    #     pydantic_model: Optional[BaseModel] = None,
    #     **kwargs
    # ) -> Tuple[AIMessage, float]:
    #     if self.should_use_azure_openai():
    #         raise AttributeError
    #     if pydantic_model == None:
    #         chat_model = self.load_model().bind(**kwargs)
    #         with get_openai_callback() as cb:
    #             res = await chat_model.ainvoke(prompt)
    #         return res, cb.total_cost

    def should_use_azure_openai(self):
        value = KEY_FILE_HANDLER.fetch_data(KeyValues.USE_AZURE_OPENAI)
        return value.lower() == "yes" if value is not None else False

    def get_model_name(self):
        if self.should_use_azure_openai():
            return "azure openai"
        elif self.model_name:
            return self.model_name

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
    ToxicityMetric)
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
        input="What is this",
        actual_output="this is a latte",
        expected_output="this is a mocha",
        retrieval_context=["I love coffee"],
        context=["I love coffee"],
    )

answer_relevancy = AnswerRelevancyMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
answer_relevancy_confine = AnswerRelevancyMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
bias = BiasMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
bias_confine = BiasMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
contextual_precision = ContextualPrecisionMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
contextual_precision_confine = ContextualPrecisionMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
contextual_recall = ContextualRecallMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
contextual_recall_confine = ContextualRecallMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
contextual_relevancy = ContextualRelevancyMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
contextual_relevancy_confine = ContextualRelevancyMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
faithfulness_recall = FaithfulnessMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
faithfulness_recall_confine = FaithfulnessMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
hallucination_recall = HallucinationMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
hallucination_recall_confine = HallucinationMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
summarization_recall = SummarizationMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
summarization_recall_confine = SummarizationMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
toxicity_recall = ToxicityMetric(verbose_mode=True, threshold=0.5, model="gpt-3.5-turbo-0125")
toxicity_recall_confine = ToxicityMetric(verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
g_eval_recall = GEval(
    name="coherence",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=['Response should be concise', 'Reponse should read easily'], 
    verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))
g_eval_recall_confine = GEval(
    name="coherence",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=['Response should be concise', 'Reponse should read easily'], 
    verbose_mode=True, threshold=0.5, model=GPTPydanticModel("gpt-3.5-turbo-0125"))


##### MEASURE

# answer_relevancy.measure(test_case)
# answer_relevancy_confine.measure(test_case)
# bias.measure(test_case)
# bias_confine.measure(test_case)
# contextual_recall.measure(test_case)
# contextual_recall_confine.measure(test_case)
# contextual_relevancy.measure(test_case)
# contextual_relevancy_confine.measure(test_case)
# faithfulness_recall.measure(test_case)
# faithfulness_recall_confine.measure(test_case)
# hallucination_recall.measure(test_case)
# hallucination_recall_confine.measure(test_case)
# summarization_recall.measure(test_case)
# summarization_recall_confine.measure(test_case)
# toxicity_recall.measure(test_case)
# toxicity_recall_confine.measure(test_case)
# g_eval_recall.measure(test_case)
# g_eval_recall_confine.measure(test_case)

##### A_MEASURE

import asyncio

async def test_answer_relevancy():
    result = await answer_relevancy.a_measure(test_case)
    print("Answer Relevancy:", result)

async def test_answer_relevancy_confine():
    result = await answer_relevancy_confine.a_measure(test_case)
    print("Answer Relevancy Confine:", result)

async def test_bias():
    result = await bias.a_measure(test_case)
    print("Bias:", result)

async def test_bias_confine():
    result = await bias_confine.a_measure(test_case)
    print("Bias Confine:", result)

async def test_contextual_precision():
    result = await contextual_precision.a_measure(test_case)
    print("Contextual Precision:", result)

async def test_contextual_precision_confine():
    result = await contextual_precision_confine.a_measure(test_case)
    print("Contextual Precision Confine:", result)

async def test_contextual_recall():
    result = await contextual_recall.a_measure(test_case)
    print("Contextual Recall:", result)

async def test_contextual_recall_confine():
    result = await contextual_recall_confine.a_measure(test_case)
    print("Contextual Recall Confine:", result)

async def test_contextual_relevancy():
    result = await contextual_relevancy.a_measure(test_case)
    print("Contextual Relevancy:", result)

async def test_contextual_relevancy_confine():
    result = await contextual_relevancy_confine.a_measure(test_case)
    print("Contextual Relevancy Confine:", result)

async def test_faithfulness_recall():
    result = await faithfulness_recall.a_measure(test_case)
    print("Faithfulness Recall:", result)

async def test_faithfulness_recall_confine():
    result = await faithfulness_recall_confine.a_measure(test_case)
    print("Faithfulness Recall Confine:", result)

async def test_hallucination_recall():
    result = await hallucination_recall.a_measure(test_case)
    print("Hallucination Recall:", result)

async def test_hallucination_recall_confine():
    result = await hallucination_recall_confine.a_measure(test_case)
    print("Hallucination Recall Confine:", result)

async def test_summarization_recall():
    result = await summarization_recall.a_measure(test_case)
    print("Summarization Recall:", result)

async def test_summarization_recall_confine():
    result = await summarization_recall_confine.a_measure(test_case)
    print("Summarization Recall Confine:", result)

async def test_toxicity_recall():
    result = await toxicity_recall.a_measure(test_case)
    print("Toxicity Recall:", result)

async def test_toxicity_recall_confine():
    result = await toxicity_recall_confine.a_measure(test_case)
    print("Toxicity Recall Confine:", result)

async def test_g_eval_recall():
    result = await g_eval_recall.a_measure(test_case)
    print("G Eval Recall:", result)

async def test_g_eval_recall_confine():
    result = await g_eval_recall_confine.a_measure(test_case)
    print("G Eval Recall Confine:", result)

async def main():
    await asyncio.gather(
        test_answer_relevancy(),
        test_answer_relevancy_confine(),
        test_bias(),
        test_bias_confine(),
        test_contextual_precision(),
        test_contextual_precision_confine(),
        test_contextual_recall(),
        test_contextual_recall_confine(),
        test_contextual_relevancy(),
        test_contextual_relevancy_confine(),
        test_faithfulness_recall(),
        test_faithfulness_recall_confine(),
        test_hallucination_recall(),
        test_hallucination_recall_confine(),
        test_summarization_recall(),
        test_summarization_recall_confine(),
        test_toxicity_recall(),
        test_toxicity_recall_confine(),
        test_g_eval_recall(),
        test_g_eval_recall_confine()
    )

# Run the main function
asyncio.run(main())
