from typing import List
from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    observe,
    update_current_span,
    RetrieverAttributes,
    LlmAttributes,
)


# Tool
@observe(type="tool")
def web_search(query: str) -> str:
    # <--Include implementation to search web here-->
    return "Latest search results for: " + query


# Retriever
@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query: str) -> List[str]:
    # <--Include implementation to fetch from vector database here-->
    fetched_documents = [
        "Document 1: This is relevant information about the query.",
        "Document 2: More relevant information here.",
        "Document 3: Additional context that might be useful.",
    ]

    update_current_span(
        attributes=RetrieverAttributes(
            embedding_input=query, retrieval_context=fetched_documents
        )
    )
    return fetched_documents


# LLM
@observe(type="llm", model="gpt-4")
def generate_response(input: str) -> str:
    # <--Include format prompts and call your LLM provider here-->
    output = "Generated response based on the prompt: " + input

    update_current_span(attributes=LlmAttributes(input=input, output=output))
    return output


# Custom span wrapping the RAG pipeline
@observe(
    type="custom",
    name="RAG Pipeline",
    metrics=["Answer Relevancy", "Faithfulness", "Contextual Relevancy"],
)
def rag_pipeline(query: str) -> str:
    # Retrieve
    docs = retrieve_documents(query)
    context = "\n".join(docs)

    # Generate
    response = generate_response(f"Context: {context}\nQuery: {query}")

    # Set test case to evaluate current span
    update_current_span(
        test_case=LLMTestCase(
            input=query, actual_output=response, retrieval_context=docs
        )
    )
    return response


# Agent that does RAG + tool calling
@observe(type="agent", available_tools=["web_search"])
def research_agent(query: str) -> str:
    # Call RAG pipeline
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


# Calling the agent will trace & trigger
# online metrics on Confident AI
research_agent("What is the weather like in San Francisco?")
