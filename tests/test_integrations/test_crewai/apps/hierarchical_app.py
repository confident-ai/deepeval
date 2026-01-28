"""
tests/test_integrations/test_crewai/apps/hierarchical_app.py
A hierarchical crew to test manager delegation traces.
"""

from crewai import Agent, Task, Crew, Process, LLM


def get_hierarchical_app():
    llm = LLM(model="gpt-4o-mini", temperature=0)

    # Junior agents
    researcher = Agent(
        role="Researcher",
        goal="Find facts",
        backstory="Junior researcher",
        llm=llm,
    )

    writer = Agent(
        role="Writer", goal="Write content", backstory="Junior writer", llm=llm
    )

    task = Task(
        description="Research frogs and write a short poem about them.",
        expected_output="A poem about frogs.",
        agent=writer,  # In hierarchical, manager delegates, but we assign a primary
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )

    return crew
