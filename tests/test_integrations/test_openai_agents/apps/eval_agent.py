"""
Eval OpenAI Agent
Complexity: MEDIUM - Uses DeepEvalAgent for metric collection
"""

from agents import ModelSettings
from deepeval.openai_agents import Agent as DeepEvalAgent

# Use DeepEvalAgent to test metric_collection passing
agent = DeepEvalAgent(
    name="EvalAgent",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.0),
    llm_metric_collection="test_llm_metrics",
    agent_metric_collection="test_agent_metrics",
)
