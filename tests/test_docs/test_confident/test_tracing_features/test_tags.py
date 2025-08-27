from deepeval.tracing import observe, update_current_trace
from openai import OpenAI

client = OpenAI()


@observe(type="agent")
def llm_app(query: str):
    update_current_trace(tags=["Causal Chit-Chat"])

    return (
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
        .choices[0]
        .message.content
    )


llm_app("Write me a poem.")
