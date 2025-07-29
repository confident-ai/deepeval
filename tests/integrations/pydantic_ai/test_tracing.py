import os
import time
from deepeval.integrations.pydantic_ai import setup_instrumentation
from pydantic_ai import Agent

setup_instrumentation(api_key="<your-api-key>")

os.environ['OPENAI_API_KEY'] = '<your-api-key>'

Agent.instrument_all()

agent = Agent(  
    'openai:gpt-4o-mini',
    system_prompt='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)

time.sleep(10)
