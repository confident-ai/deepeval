import os
import time
from dotenv import load_dotenv
from deepeval.integrations.pydantic_ai import setup_instrumentation
from deepeval.integrations.pydantic_ai import Agent

load_dotenv()

setup_instrumentation(api_key=os.getenv("CONFIDENT_API_KEY"))

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

Agent.instrument_all()

agent = Agent(  
    'openai:gpt-4o-mini',
    system_prompt='Be concise, reply with one sentence.',  
    metric_collection='test_collection_1',
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)

time.sleep(10)
