"""
tests/test_integrations/test_crewai/apps/evals_app.py
A crew designed to test DeepEval specific attributes like metric_collection
propagation across Traces, Crews, Agents, and Tools.
"""

from crewai import Task
from deepeval.integrations.crewai import Crew, Agent, LLM, tool


@tool(metric_collection="tool_metrics_v1")
def special_metric_tool(query: str) -> str:
    """A tool that claims to calculate special metrics."""
    return f"Calculated metrics for: {query}"


def get_evals_crew():
    llm = LLM(model="gpt-4o-mini", temperature=0)

    agent = Agent(
        role="Evaluator",
        goal="Validate metrics",
        backstory="You ensure all metadata is correctly attached. You are robotic and precise.",
        llm=llm,
        tools=[special_metric_tool],
        metric_collection="agent_metrics_v1",
        verbose=True,
    )

    task = Task(
        description="Use the special_metric_tool to process '{input}'. Your final answer MUST be EXACTLY the output of the tool. Do not add any other text.",
        expected_output="The exact string returned by the tool.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        metric_collection="crew_metrics_v1",
        verbose=True,
    )

    return crew
