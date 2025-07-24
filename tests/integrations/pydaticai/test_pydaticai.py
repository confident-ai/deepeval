import os
import time
from deepeval.integrations.pydantic_ai.agent import PydanticAIAgent as Agent
from deepeval.integrations.pydantic_ai import setup_instrumentation

setup_instrumentation(api_key="<your-api-key>")

os.environ['OPENAI_API_KEY'] = '<your-api-key>'

Agent.instrument_all()

agent = Agent(  
    'openai:gpt-4o-mini',
    system_prompt='Be concise, reply with one sentence.',  
    metric_collection="test_collection_1"
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)

time.sleep(10)

