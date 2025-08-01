import os
import time
from deepeval.integrations.pydantic_ai import setup_instrumentation
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

setup_instrumentation(api_key=os.getenv("CONFIDENT_API_KEY"))

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

Agent.instrument_all()

# this will not create `confident agent run` as it is not using `deepeval.integrations.pydantic_ai import Agent`
# it will just create spans sent by pydantic ai to the ConfidentExporter
agent = Agent(  
    'openai:gpt-4o-mini',
    system_prompt='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)

time.sleep(10)
