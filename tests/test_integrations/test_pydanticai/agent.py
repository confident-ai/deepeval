import asyncio
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import ConfidentInstrumentationSettings

confident_instrumentation_settings = ConfidentInstrumentationSettings()

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    instrument=confident_instrumentation_settings,
)
    
async def run_agent():
    confident_instrumentation_settings.thread_id = "test_thread_id_1"
    result = await agent.run("What are the LLMs?")
    print(result)

asyncio.run(run_agent())