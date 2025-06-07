import time
from openai import OpenAI

import deepeval
from deepeval.tracing import observe, trace_manager

deepeval.login_with_confident_api_key("<your-confident-api-key>")

# Initialize OpenAI client
client = OpenAI(api_key="<your-openai-api-key>")

trace_manager.configure(openai_client=client)

@observe(type="llm")
def generate_response(input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or your preferred model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ],
        temperature=0.7,
    )

    # response = client.beta.chat.completions.parse(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": input}
    #     ],
    # )
    return response.choices[0].message.content


try:
    response = generate_response("What is the weather in Tokyo?")
    print(response)
except Exception as e:
    raise e

time.sleep(8)
