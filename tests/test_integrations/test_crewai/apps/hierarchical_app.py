"""
tests/test_integrations/test_crewai/apps/hierarchical_app.py
A hierarchical crew to test manager delegation traces.
"""

from crewai import Agent, Task, Crew, Process, LLM


def get_hierarchical_app():
    llm = LLM(model="gpt-4o-mini", temperature=0)


    writer = Agent(
        role="Writer",
        goal="Write simple words",
        backstory="You are a junior writer. You simply write back what you are told.",
        llm=llm,
        verbose=True,
    )

    task = Task(
        description="Manager: You must delegate a task to the 'Writer' agent to write the word: 'SUCCESS'. Do nothing else.",
        expected_output="The word 'SUCCESS'.",
        agent=writer,  # In hierarchical, this is the target, but Manager orchestrates
    )

    crew = Crew(
        agents=[writer],  # Only 1 worker needed to test delegation tracing
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )

    return crew
