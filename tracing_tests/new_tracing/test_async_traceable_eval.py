from asyncio import sleep
import random

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    FaithfulnessMetric,
    TaskCompletionMetric,
)
from deepeval.tracing import (
    update_current_span,
    observe,
    update_llm_span,
    update_current_trace,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    BinaryJudgementNode,
    VerdictNode,
)
from deepeval.metrics import DAGMetric, GEval
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

geval_metric = GEval(
    name="Persuasiveness",
    criteria="Determine how persuasive the `actual output` is to getting a user booking in a call.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

conciseness_node = BinaryJudgementNode(
    criteria="Does the actual output contain less than or equal to 4 sentences?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=geval_metric),
    ],
)

# create the DAG
dag = DeepAcyclicGraph(root_nodes=[conciseness_node])
metric = DAGMetric(dag=dag, name="DAG")


#######################################################
## Example ############################################
#######################################################


@observe(type="llm", model="gpt-4o")
async def generate_text(prompt: str):
    generated_text = f"Generated text for: {prompt}"
    update_current_span(
        input=prompt,
        output=generated_text,
    )
    update_llm_span(
        input_token_count=len(prompt.split()),
        output_token_count=len(generated_text.split()),
    )
    await sleep(random.uniform(1, 3))
    return generated_text


# Example of a retrieval node with embedded embedder
@observe(type="retriever", embedder="text-embedding-ada-002")
async def retrieve_documents(query: str, top_k: int = 3):
    documents = [
        f"Document 1 about {query}",
        f"Document 2 about {query}",
        f"Document 3 about {query}",
    ]
    update_current_span(
        input=query,
        retrieval_context=documents,
    )
    return documents


@observe("CustomEmbedder")
async def custom_embed(text: str, model: str = "custom-model"):
    embedding = [0.1, 0.2, 0.3]
    await sleep(random.uniform(1, 3))
    return embedding


@observe("CustomRetriever", name="custom retriever")
async def custom_retrieve(query: str, embedding_model: str = "custom-model"):
    embedding = await custom_embed(query, embedding_model)
    documents = [
        f"Custom doc 1 about {query}",
        f"Custom doc 2 about {query}",
    ]
    await sleep(random.uniform(1, 3))


@observe("CustomLLM")
async def custom_generate(prompt: str, model: str = "custom-model"):
    response = f"Custom response for: {prompt}"
    await sleep(random.uniform(1, 3))
    return response


@observe(type="agent", available_tools=["custom_retrieve", "custom_generate"])
async def custom_research_agent(query: str):
    if random.random() < 0.5:
        docs = await custom_retrieve(query)
        analysis = await custom_generate(str(docs))
        await sleep(random.uniform(1, 3))
        return analysis
    else:
        await sleep(random.uniform(1, 3))
        return "Research information unavailable"


@observe(
    available_tools=["get_weather", "get_location"],
    metrics=[TaskCompletionMetric()],
)
async def weather_agent(query: str):
    update_current_span(
        test_case=LLMTestCase(
            input=query, actual_output="Weather information unavailable"
        )
    )
    await sleep(random.uniform(1, 3))
    return "Weather information unavailable"


@observe(type="agent", available_tools=["retrieve_documents", "generate_text"])
async def research_agent(query: str):
    if random.random() < 0.5:
        docs = await retrieve_documents(query)
        analysis = await generate_text(str(docs))
        # Add sleep of 1-3 seconds
        await sleep(random.uniform(1, 3))
        return analysis
    else:
        # Add sleep of 1-3 seconds
        await sleep(random.uniform(1, 3))
        return "Research information unavailable"


@observe(
    type="agent",
    agent_handoffs=["research_agent", "custom_research_agent"],
    metrics=[TaskCompletionMetric(task="Get the weather"), geval_metric],
    metric_collection="Test",
)
async def meta_agent(input: str):
    # 50% probability of executing the function
    weather_info = await weather_agent(input)
    research_info = await research_agent(input)
    custom_info = await custom_research_agent(input)
    final_response = f"""
    Weather: {weather_info}
    Research: {research_info}
    Custom Analysis: {custom_info}
    """

    update_current_span(
        test_case=LLMTestCase(input=input, actual_output=final_response)
    )
    return final_response


@observe(
    type="agent",
    agent_handoffs=["research_agent", "custom_research_agent"],
    metrics=[TaskCompletionMetric(task="Get the weather"), geval_metric],
)
def sync_meta_agent(input: str):
    final_response = f"""
    Weather: {input}
    Research: {input}
    Custom Analysis: {input}
    """

    update_current_span(
        test_case=LLMTestCase(input=input, actual_output=final_response)
    )
    update_current_trace(input=input, output=final_response)
    return final_response


###################################v

import asyncio
from deepeval.dataset import Golden, EvaluationDataset
from deepeval import evaluate

goldens = [
    Golden(input="What's the weather like in SF?"),
    # Golden(input="Tell me about Elon Musk."),
    # Golden(input="What's the weather like in SF?"),
    # Golden(input="Tell me about Elon Musk."),
    # Golden(input="What's the weather like in SF?"),
    # Golden(input="Tell me about Elon Musk."),
    # Golden(input="What's the weather like in SF?"),
    # Golden(input="Tell me about Elon Musk."),
    # Golden(input="What's the weather like in SF?"),
    # Golden(input="Tell me about Elon Musk."),
]

dataset = EvaluationDataset(goldens=goldens)

from deepeval.metrics import AnswerRelevancyMetric

# for golden in dataset.evals_iterator(metrics=[AnswerRelevancyMetric()]):
#     task = asyncio.create_task(meta_agent(golden.input))
#     dataset.evaluate(task)


for golden in dataset.evals_iterator(
    metrics=[AnswerRelevancyMetric()], async_config=AsyncConfig(run_async=False)
):
    sync_meta_agent(golden.input)


# # Run Async
#    goldens=goldens * 40,
#    observed_callback=meta_agent,
#    async_config=AsyncConfig(run_async=True, max_concurrent=40),
#    cache_config=CacheConfig(write_cache=False),
#    # display_config=DisplayConfig(show_indicator=False),

# evaluate(
#     goldens=goldens,
#     observed_callback=meta_agent,
# async_config=AsyncConfig(run_async=True, max_concurrent=40),
# cache_config=CacheConfig(write_cache=False),
# display_config=DisplayConfig(show_indicator=False),
# )
# evaluate(
#     goldens=goldens,
#     observed_callback=meta_agent,
#     async_config=AsyncConfig(run_async=True),
#     display_config=DisplayConfig(show_indicator=False),
# )
# evaluate(
#     goldens=goldens,
#     observed_callback=meta_agent,
#     async_config=AsyncConfig(run_async=True),
#     display_config=DisplayConfig(show_indicator=True),
# )

# import pytest


# # Assert Test
# @pytest.mark.parametrize(
#     "golden",
#     goldens,
# )
# def test_meta_agent_0(golden):
#     assert_test(golden=golden, observed_callback=meta_agent)

# # Gather multiple traceable tasks
# import asyncio


# async def run_parallel_examples():
#     tasks = [
#         meta_agent("How tall is Mount Everest?"),
#         meta_agent("What's the capital of Brazil?"),
#         meta_agent("Who won the last World Cup?"),
#         meta_agent("Explain quantum entanglement."),
#         meta_agent("What's the latest iPhone model?"),
#         meta_agent("How do I cook a perfect steak?"),
#         meta_agent("Tell me a joke about robots."),
#         meta_agent("What causes lightning?"),
#         meta_agent("Who painted the Mona Lisa?"),
#         meta_agent("What's the population of Japan?"),
#         meta_agent("How do vaccines work?"),
#         meta_agent("Recommend a good sci-fi movie."),
#     ]
#     await asyncio.gather(*tasks)


# # # Run it
# asyncio.run(run_parallel_examples())
