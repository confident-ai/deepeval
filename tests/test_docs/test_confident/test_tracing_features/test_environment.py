from openai import OpenAI
from deepeval.tracing import observe, trace_manager

trace_manager.configure(environment="production")
client = OpenAI()


@observe()
def llm_app(query: str):
    return (
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
        .choices[0]
        .message.content
    )


llm_app("Write me a poem.")
