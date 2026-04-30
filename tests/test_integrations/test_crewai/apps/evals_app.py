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
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=50,
    )

    agent = Agent(
        role="ToolCaller",
        goal="Call the special_metric_tool exactly once and return its raw output.",
        backstory="You MUST call the provided tool exactly once. You MUST NOT reason. You MUST NOT retry. You MUST NOT modify the output.",
        llm=llm,
        tools=[special_metric_tool],
        metric_collection="agent_metrics_v1",
        allow_delegation=False,
        verbose=False,
        max_iter=1,
    )

    task = Task(
        description=(
            "STRICT INSTRUCTIONS:\n"
            "1. You MUST call special_metric_tool exactly once.\n"
            "2. Use the input string exactly as provided.\n"
            "3. Your FINAL ANSWER must be EXACTLY the tool's raw output.\n"
            "4. Do not add any commentary.\n"
            "5. Do not explain anything.\n"
        ),
        expected_output="Calculated metrics for: deterministic_test_input",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        metric_collection="crew_metrics_v1",
        verbose=False,
        max_iter=1,
        process="sequential",
    )

    return crew
