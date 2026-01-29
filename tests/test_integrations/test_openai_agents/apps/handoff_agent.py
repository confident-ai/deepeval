"""
Handoff OpenAI Agent
Complexity: HIGH - Multi-agent swarm
"""
from agents import Agent, ModelSettings

spanish = Agent(
    name="SpanishAgent",
    instructions="You speak Spanish. Answer 'Hola' to everything.",
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.0),
)

english = Agent(
    name="EnglishAgent",
    instructions="You speak English. Answer 'Hello' to everything.",
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.0),
)

triage_agent = Agent(
    name="TriageAgent",
    instructions="If input is Spanish, handoff to SpanishAgent. Else EnglishAgent.",
    model="gpt-4o",
    handoffs=[spanish, english],
    model_settings=ModelSettings(temperature=0.0),
)