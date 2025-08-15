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

    update_current_trace(user_id="your-user-id")
    return res


llm_app("Write me a poem.")
