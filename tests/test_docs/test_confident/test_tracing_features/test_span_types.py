from deepeval.tracing import observe


@observe(name="RAG Pipeline")
def rag_pipeline(query: str) -> str:
    pass


############################################

from deepeval.tracing import observe


@observe(type="llm", model="gpt-4")
def generate_response(prompt: str) -> str:
    pass


############################################

from deepeval.tracing import observe
from typing import List


@observe(type="retriever", embedder="text-embedding-ada-002")
def retrieve_documents(query: str) -> List[str]:
    pass


############################################

from deepeval.tracing import observe


@observe(type="tool")
def web_search(query: str) -> str:
    pass


from deepeval.tracing import observe

############################################

from deepeval.tracing import observe


@observe(
    type="agent",
    available_tools=["search", "calculator"],
    handoff_agents=["research_agent", "math_agent"],
)
def supervisor_agent(query: str) -> str:
    pass


############################################

from typing import List
from deepeval.tracing import (
    observe,
    update_current_span,
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
        input=query,
        retrieval_context=fetched_documents,
    )
    return fetched_documents


# LLM
@observe(type="llm", model="gpt-4")
def generate_response(input: str) -> str:
    # <--Include format prompts and call your LLM provider here-->
    output = "Generated response based on the prompt: " + input
    update_current_span(input=input, output=output)
    return output


# Custom span wrapping the RAG pipeline
@observe(type="custom", name="RAG Pipeline")
def rag_pipeline(query: str) -> str:
    # Retrieve
    docs = retrieve_documents(query)
    context = "\n".join(docs)

    # Generate
    response = generate_response(f"Context: {context}\nQuery: {query}")
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


# Calling the agent will create traces on Confident AI
research_agent("What is the weather like in San Francisco?")
