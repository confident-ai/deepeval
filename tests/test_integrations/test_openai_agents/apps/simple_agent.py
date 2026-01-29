"""
Simple OpenAI Agent
Complexity: LOW - Standard Agent, no tools
"""
from agents import Agent, ModelSettings


agent = Agent(
    name="SimpleAgent",
    instructions="You are a helpful assistant. Answer the user's question concisely. Do not use any tools.",
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.0),
)