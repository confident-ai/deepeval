from typing import List

from deepeval.metrics.contextual_relevancy.contextual_relevancy import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    observe,
    update_current_span_test_case,
)


@observe(type="tool")
def web_search(query: str) -> str:
    # <--Include implementation to search web here-->
    return "Latest search results for: " + query


@observe(type="retriever")
def retrieve_documents(query: str) -> List[str]:
    # <--Include implementation to fetch from vector database here-->
    return ["Document 1: This is relevant information about the query."]


@observe(type="llm")
def generate_response(input: str) -> str:
    # <--Include format prompts and call your LLM provider here-->
    return "Generated response based on the prompt: " + input


@observe(
    type="custom", name="RAG Pipeline", metrics=[ContextualRelevancyMetric()]
)
def rag_pipeline(query: str) -> str:
    # Calls retriever and llm
    docs = retrieve_documents(query)
    context = "\n".join(docs)
    response = generate_response(f"Context: {context}\nQuery: {query}")

    update_current_span_test_case(
        test_case=LLMTestCase(
            input=query, actual_output=response, retrieval_context=docs
        )
    )
    return response


@observe(type="agent", available_tools=["web_search"])
def research_agent(query: str) -> str:
    # Calls RAG pipeline
    initial_response = rag_pipeline(query)

    # Use web search tool on the results
    search_results = web_search(initial_response)

    # Generate final response incorporating both RAG and search results
    final_response = generate_response(
        f"Initial response: {initial_response}\n"
        f"Additional search results: {search_results}\n"
        f"Query: {query}"
    )
    return final_response


research_agent("What is the weather like in San Francisco?")
