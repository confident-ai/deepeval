from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import observe, update_current_span
from openai import OpenAI

client = OpenAI()


@observe(metrics=[AnswerRelevancyMetric()])
def complete(query: str):
    response = (
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
        .choices[0]
        .message.content
    )

    update_current_span(test_case=LLMTestCase(input=query, output=response))
    return response


################################

from typing import List

from deepeval.test_case import LLMTestCase
from deepeval.tracing import (
    observe,
    update_current_span,
)
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden


def web_search(query: str) -> str:
    return "Fake search results for: " + query


def retrieve_documents(query: str) -> List[str]:
    return ["Document 1: Hardcoded text chunks from your vector DB"]


@observe(metrics=[AnswerRelevancyMetric()])
def generate_response(input: str) -> str:
    response = "Generated response based on the prompt: " + input

    update_current_span(
        test_case=LLMTestCase(input=input, actual_output=response)
    )
    return response


@observe(name="RAG Pipeline", metrics=[ContextualRelevancyMetric()])
def rag_pipeline(query: str) -> str:
    # Calls retriever and llm
    docs = retrieve_documents(query)
    context = "\n".join(docs)
    response = generate_response(f"Context: {context}\nQuery: {query}")

    update_current_span(
        test_case=LLMTestCase(
            input=query, actual_output=response, retrieval_context=docs
        )
    )
    return response


@observe(type="agent")
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


from deepeval.dataset import Golden
from deepeval import evaluate

# Create golden instead of test case
golden = Golden(input="What's the weather like in SF?")

# Run evaluation
dataset = EvaluationDataset(goldens=[golden])
for golden in dataset.evals_iterator():
    research_agent(golden.input)
