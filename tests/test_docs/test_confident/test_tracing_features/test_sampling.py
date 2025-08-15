from deepeval.tracing import observe, trace_manager
from openai import OpenAI

client = OpenAI()
trace_manager.configure(sampling_rate=0.5)


@observe()
def llm_app(query: str):
    return (
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
        .choices[0]
        .message.content
    )


for _ in range(10):
    llm_app("Write me a poem.")  # roughly half of these traces will be sent
