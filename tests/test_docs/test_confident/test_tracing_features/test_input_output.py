from openai import OpenAI
from deepeval.tracing import observe, update_current_trace

client = OpenAI()


@observe()
def llm_app(query: str):
    res = (
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
        .choices[0]
        .message.content
    )

    update_current_trace(input=query, output=res)
    return res


llm_app("Write me a poem.")

############################################

from openai import OpenAI
from deepeval.tracing import observe, update_current_span

client = OpenAI()


@observe()
def llm_app(query: str):
    res = (
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
        .choices[0]
        .message.content
    )

    update_current_span(input=query, output=res)
    return res


llm_app("Write me a poem.")
