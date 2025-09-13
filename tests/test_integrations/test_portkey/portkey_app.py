import os
from dotenv import load_dotenv
from portkey_ai import Portkey

load_dotenv()

from deepeval.integrations.portkey import instrument    
instrument()

config = {
    "provider": 'openai',
    "api_key": os.getenv("PORTKEY_API_KEY")
}

client = Portkey(config = config, metric_collection="test_collection_1")
print(client)

# Example: Send a chat completion request
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    model="gpt-4o"
)

print(response.choices[0].message.content)