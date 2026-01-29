"""
tests/test_integrations/test_openai_agents/apps/evals_app.py
Tests Metric Collections (Agent/LLM/Tool scopes) and Prompt Logging.
"""

from deepeval.openai_agents import Agent, function_tool
from deepeval.prompt import Prompt


@function_tool(metric_collection="tool_evals")
def calculate_score(input: str) -> str:
    """Returns a dummy score."""
    return "Score: 100"


def get_evals_app():
    prompt = Prompt(alias="test-prompt", text_template="You are a helper.")
    prompt._version = "00.00.01"

    agent = Agent(
        name="Eval Agent",
        instructions="Use the tool to get the score.",
        tools=[calculate_score],
        agent_metric_collection="agent_evals",
        llm_metric_collection="llm_evals",
        confident_prompt=prompt,
        tool_use_behavior="run_llm_again",
    )

    return agent, "Get the score."
