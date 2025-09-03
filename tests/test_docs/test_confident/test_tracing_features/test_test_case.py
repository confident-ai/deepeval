from deepeval.tracing import observe, update_current_span
from deepeval.test_case import ToolCall


@observe()
def llm_app(query: str):
    update_current_span(
        input=query,
        output="LLM app response",
        tools_called=[
            ToolCall(name="web_search", input_parameters={"query": query})
        ],
    )
    return "LLM app response"


llm_app("What is weather in San Francisco?")

############################################

from openai import OpenAI
from deepeval.tracing import observe, update_current_trace

client = OpenAI()


@observe()
def retriever_component(query: str):
    retrieved_chunks = ["chunk1", "chunk2"]
    update_current_trace(retrieval_context=retrieved_chunks)
    return "\n".join(retrieved_chunks)


@observe()
def llm_app(query: str):
    retrieval_context = retriever_component(query)
    res = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query + retrieval_context}],
        )
        .choices[0]
        .message.content
    )
    update_current_trace(input=query, output=res)
    return res


llm_app("What is weather typically like in San Francisco?")
