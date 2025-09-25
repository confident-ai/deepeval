from deepeval.openai_agents import Agent, Runner
from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    confident_prompt=prompt,
    llm_metric_collection="test_collection_1",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)