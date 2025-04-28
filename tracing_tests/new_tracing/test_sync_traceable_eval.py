from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    update_current_span_test_case,
    update_current_span_attributes,
    observe,
    RetrieverAttributes,
    LlmAttributes,
    trace_manager,
)

from time import sleep, perf_counter
import random

trace_manager._daemon = False

#######################################################
## Example ############################################
#######################################################


@observe(type="llm", model="gpt-4o")
def generate_text(prompt: str):
    generated_text = f"Generated text for: {prompt}"
    attributes = LlmAttributes(
        input=prompt,
        output=generated_text,
        input_token_count=len(prompt.split()),
        output_token_count=len(generated_text.split()),
    )
    # update_current_span_attributes(attributes)
    sleep(random.uniform(1, 3))
    return generated_text


# Example of a retrieval node with embedded embedder
@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query: str, top_k: int = 3):
    documents = [
        f"Document 1 about {query}",
        f"Document 2 about {query}",
        f"Document 3 about {query}",
    ]
    # update_current_span_attributes(
    #     RetrieverAttributes(
    #         embedding_input=query,
    #         retrieval_context=documents,
    #     )
    # )
    return documents


@observe("CustomEmbedder")
def custom_embed(text: str, model: str = "custom-model"):
    embedding = [0.1, 0.2, 0.3]
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return embedding


@observe("CustomRetriever", name="custom retriever")
def custom_retrieve(query: str, embedding_model: str = "custom-model"):
    embedding = custom_embed(query, embedding_model)
    documents = [
        f"Custom doc 1 about {query}",
        f"Custom doc 2 about {query}",
    ]
    sleep(random.uniform(1, 3))


@observe("CustomLLM")
def custom_generate(prompt: str, model: str = "custom-model"):
    response = f"Custom response for: {prompt}"
    sleep(random.uniform(1, 3))
    return response


@observe(type="agent", available_tools=["custom_retrieve", "custom_generate"])
def custom_research_agent(query: str):
    if random.random() < 0.5:
        docs = custom_retrieve(query)
        analysis = custom_generate(str(docs))
        sleep(random.uniform(1, 3))
        return analysis
    else:
        sleep(random.uniform(1, 3))
        return "Research information unavailable"


@observe(
    type="agent",
    available_tools=["get_weather", "get_location"],
    metrics=[AnswerRelevancyMetric(), BiasMetric()],
)
def weather_agent(query: str):
    update_current_span_test_case(
        LLMTestCase(
            input=query, actual_output="Weather information unavailable"
        )
    )
    sleep(random.uniform(1, 3))
    return "Weather information unavailable"


@observe(type="agent", available_tools=["retrieve_documents", "generate_text"])
def research_agent(query: str):
    docs = retrieve_documents(query)
    analysis = generate_text(str(docs))
    # Add sleep of 1-3 seconds
    sleep(random.uniform(1, 3))
    return analysis


@observe(
    type="agent",
    agent_handoffs=["weather_agent", "research_agent", "custom_research_agent"],
    metrics=[AnswerRelevancyMetric(), BiasMetric()],
)
def meta_agent(input: str):
    # 50% probability of executing the function
    weather_info = weather_agent(input)
    research_info = research_agent(input)
    custom_info = custom_research_agent(input)
    final_response = f"""
    Weather: {weather_info}
    Research: {research_info}
    Custom Analysis: {custom_info}
    """
    update_current_span_test_case(
        test_case=LLMTestCase(input=input, actual_output=final_response)
    )
    return final_response


###################################v

from deepeval.dataset import Golden
from deepeval import evaluate, assert_test

goldens = [
    Golden(input="What's the weather like in SF?"),
    Golden(input="Tell me about Elon Musk."),
]

# # Run Async
# evaluate(goldens, meta_agent, run_async=True, show_indicator=True)
# evaluate(goldens, meta_agent, run_async=True, show_indicator=False)

# # Run Sync
# evaluate(goldens, meta_agent, run_async=False, show_indicator=True)
# evaluate(goldens, meta_agent, run_async=False, show_indicator=False)


# Assert Test
def test_meta_agent_0():
    golden = Golden(input="What's the weather like in SF?")
    assert_test(golden=golden, traceable_callback=meta_agent, run_async=False)


def test_meta_agent_1():
    golden = Golden(input="What's the weather like in SF?")
    assert_test(golden=golden, traceable_callback=meta_agent, run_async=False)
