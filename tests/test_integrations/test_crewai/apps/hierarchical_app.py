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
        goal="Find facts about frogs",
        backstory="Junior researcher specialized in amphibians.",
        llm=llm,
    )

    writer = Agent(
        role="Writer",
        goal="Write content about frogs",
        backstory="Junior writer who writes poems.",
        llm=llm,
    )

    task = Task(
        description="Research frogs and write a short poem about them. You are the Manager. You MUST delegate the research task to the Researcher and the writing task to the Writer. Do not perform these tasks yourself.",
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
