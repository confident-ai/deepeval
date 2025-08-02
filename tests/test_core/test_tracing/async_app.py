from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    RetrieverAttributes,
    LlmAttributes,
    update_current_span,
    observe,
)
import random


@observe(type="llm", model="gpt-4o")
async def generate_text(prompt: str):
    generated_text = f"Generated text for: {prompt}"
    attributes = LlmAttributes(
        input=prompt,
        output=generated_text,
        input_token_count=len(prompt.split()),
        output_token_count=len(generated_text.split()),
    )
    update_current_span(attributes=attributes)
    return generated_text


@observe(type="retriever", embedder="text-embedding-ada-002")
async def retrieve_documents(query: str, top_k: int = 3):
    documents = [
        f"Document 1 about {query}",
        f"Document 2 about {query}",
        f"Document 3 about {query}",
    ]
    update_current_span(
        attributes=RetrieverAttributes(
            embedding_input=query,
            retrieval_context=documents,
        )
    )
    return documents


@observe("CustomEmbedder")
async def custom_embed(text: str, model: str = "custom-model"):
    embedding = [0.1, 0.2, 0.3]
    return embedding


@observe("CustomRetriever", name="custom retriever")
async def custom_retrieve(query: str, embedding_model: str = "custom-model"):
    embedding = await custom_embed(query, embedding_model)
    documents = [
        f"Custom doc 1 about {query}",
        f"Custom doc 2 about {query}",
    ]
    return documents


@observe("CustomLLM")
async def custom_generate(prompt: str, model: str = "custom-model"):
    response = f"Custom response for: {prompt}"
    return response


@observe(type="agent", available_tools=["custom_retrieve", "custom_generate"])
async def custom_research_agent(query: str):
    if random.random() < 0.5:
        docs = await custom_retrieve(query)
        analysis = await custom_generate(str(docs))
        return analysis
    else:
        return "Research information unavailable"


@observe(
    available_tools=["get_weather", "get_location"],
    metrics=[AnswerRelevancyMetric()],
)
async def weather_agent(query: str):
    update_current_span(
        test_case=LLMTestCase(
            input=query, actual_output="Weather information unavailable"
        )
    )
    return "Weather information unavailable"


@observe(type="agent", available_tools=["retrieve_documents", "generate_text"])
async def research_agent(query: str):
    if random.random() < 0.5:
        docs = await retrieve_documents(query)
        analysis = await generate_text(str(docs))
        return analysis
    else:
        return "Research information unavailable"


@observe(
    type="agent",
    agent_handoffs=["research_agent", "custom_research_agent"],
    metrics=[TaskCompletionMetric(task="Get the weather")],
    metric_collection="Test",
)
async def meta_agent(input: str):
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
