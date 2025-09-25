import asyncio
from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import ConfidentInstrumentationSettings
from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

confident_instrumentation_settings = ConfidentInstrumentationSettings(
    thread_id="test_thread_id_1",
    user_id="test_user_id_1",
    metadata={"test_metadata_key": "test_metadata_value"},
    tags=["test_tag_1", "test_tag_2"],
    metric_collection="test_metric_collection_1",
    environment="testing",
    name="test_name_1",
    confident_prompt=prompt,
)

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
    instrument=confident_instrumentation_settings,
    name="test_agent",
)
    
async def run_agent():
    result = await agent.run("What are the LLMs?")
    print(result)

asyncio.run(run_agent())