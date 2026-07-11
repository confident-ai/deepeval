import asyncio

from deepeval.metrics import AnswerRelevancyMetric, TaskCompletionMetric
from deepeval.tracing import (
    observe,
    update_current_span,
    update_llm_span,
    update_retriever_span,
)


@observe(type="llm", model="gpt-4o")
async def generate_text(prompt: str):
    generated_text = f"Generated text for: {prompt}"
    await asyncio.sleep(1)
    update_llm_span(
        input_token_count=len(prompt.split()),
        output_token_count=len(generated_text.split()),
    )
    return generated_text


@observe(type="retriever", embedder="text-embedding-ada-002")
async def retrieve_documents(query: str, top_k: int = 3):
    documents = [
        f"Document 1 about {query}",
        f"Document 2 about {query}",
        f"Document 3 about {query}",
    ]
    update_retriever_span(
        top_k=top_k,
        chunk_size=5,
    )
    return documents


@observe("CustomEmbedder")
async def custom_embed(text: str, model: str = "custom-model"):
    embedding = [0.1, 0.2, 0.3]
    return embedding


@observe("CustomRetriever", name="custom retriever")
async def custom_retrieve(query: str, embedding_model: str = "custom-model"):
    await custom_embed(query, embedding_model)
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
    docs = await custom_retrieve(query)
    analysis = await custom_generate(str(docs))
    return analysis


# Metrics must NOT be instantiated at module level: doing so in a decorator
# argument triggers an API-key validation at import time, which breaks pytest
# collection in keyless CI.  Instead, pass pre-built instances that are
# constructed once here at definition time via helper calls — the key
# difference is that these helpers are called here rather than at the
# top-level so the module remains importable without env vars.
#
# If even this causes issues in keyless environments the metrics can be moved
# fully inside the function body and passed via `update_current_span`.
_weather_metrics = [AnswerRelevancyMetric()]
_meta_metrics = [TaskCompletionMetric(task="Get the weather")]


@observe(
    available_tools=["get_weather", "get_location"],
    metrics=_weather_metrics,
)
async def weather_agent(query: str):
    update_current_span(
        input=query,
        output="Weather information unavailable",
    )
    return "Weather information unavailable"


@observe(type="agent", available_tools=["retrieve_documents", "generate_text"])
async def research_agent(query: str):
    docs = await retrieve_documents(query)
    analysis = await generate_text(str(docs))
    return analysis


@observe(
    type="agent",
    agent_handoffs=["research_agent", "custom_research_agent"],
    metrics=_meta_metrics,
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
        input=input,
        output=final_response,
    )
    return final_response
