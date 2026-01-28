"""
tests/test_integrations/test_crewai/apps/simple_app.py
A basic single-agent crew for testing simple kickoff traces.
"""
from crewai import Task
from crewai.tools import tool

from deepeval.integrations.crewai import Crew, Agent, LLM, tool

def get_simple_app(id_suffix: str = ""):
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0,
        metric_collection="metric_collection_1",
    )

    agent = Agent(
        role=f"Simple Greeter {id_suffix}",
        goal="Reply to greetings",
        backstory="You are a friendly bot.",
        llm=llm,
        metric_collection="metric_collection_1",
    )

    task = Task(
        description="Reply to the user: {input}",
        expected_output="A short greeting.",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        metric_collection="metric_collection_1",
        verbose=True
    )
    
    return crew