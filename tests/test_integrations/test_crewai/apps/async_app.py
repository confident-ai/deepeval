"""
tests/test_integrations/test_crewai/apps/async_app.py
A crew designed for async execution tests.
"""

from crewai import Agent, Task, Crew, LLM
import asyncio

# Mock async tool logic handled inside the task flow usually,
# but for standard CrewAI, async execution mostly happens at agent level.


def get_async_app():
    llm = LLM(model="gpt-4o-mini", temperature=0)

    agent = Agent(
        role="Async Worker",
        goal="Process requests fast",
        backstory="Digital worker",
        llm=llm,
        verbose=True,
    )

    task = Task(
        description="Process this input asynchronously: {input}",
        expected_output="Processed output.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    return crew
