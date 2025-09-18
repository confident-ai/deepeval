import os
from deepeval.integrations.portkey import Portkey, instrument_portkey
instrument_portkey()

config = {
    "provider": 'openai',
    "api_key": os.getenv("OPENAI_API_KEY")
}

client = Portkey(config = config)

def execute_chat_completion():
    # Example: Send a chat completion request
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="gpt-4o", 
        metric_collection="test_collection_1"
    )
    print(response.choices[0].message.content)

execute_chat_completion()