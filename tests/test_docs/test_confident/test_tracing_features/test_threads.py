from deepeval.tracing import observe, update_current_trace
from openai import OpenAI

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

    update_current_trace(thread_id="your-thread-id", input=query, output=res)
    return res


llm_app("Write me a poem.")

############################################

from deepeval.tracing import observe, update_current_trace
from openai import OpenAI

client = OpenAI()


@observe()
def llm_app(query: str):
    messages = {"role": "user", "content": query}
    res = (
        client.chat.completions.create(model="gpt-4o", messages=messages)
        .choices[0]
        .message.content
    )

    # ✅ Do this, query is the raw user input
    update_current_trace(thread_id="your-thread-id", input=query, output=res)

    # ❌ Don't do this, messages is not the raw user input
    # update_current_trace(thread_id="your-thread-id", input=messages, output=res)
    return res


from deepeval.tracing import observe, update_current_trace
from openai import OpenAI

client = OpenAI()
