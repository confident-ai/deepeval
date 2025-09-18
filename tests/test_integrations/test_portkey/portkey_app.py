import os
from deepeval.integrations.portkey import Portkey

from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

config = {
    "provider": 'openai',
    "api_key": os.getenv("OPENAI_API_KEY")
}

client = Portkey(config = config)

def execute_chat_completion():
    response = client.chat.completions.create(
        messages=[
            # {"role": "system", "content": prompt.interpolate(name=)},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        model="gpt-4o", 
        metric_collection="test_collection_1",
        prompt=prompt,
    )
    print(response.choices[0].message.content)

# execute_chat_completion()