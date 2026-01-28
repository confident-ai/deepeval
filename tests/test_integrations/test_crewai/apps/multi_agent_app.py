"""
tests/test_integrations/test_crewai/apps/multi_agent_app.py
A multi-agent crew (Sequential Process) to test span parentage and ordering.
"""

from crewai import Agent, Task, Crew, LLM


def get_multi_agent_app():
    llm = LLM(model="gpt-4o-mini", temperature=0)

    # Agent 1: Researcher
    researcher = Agent(
        role="Researcher",
        goal="Find a topic",
        backstory="Curious researcher",
        llm=llm,
        verbose=True,
    )

    # Agent 2: Writer
    writer = Agent(
        role="Writer",
        goal="Write a joke about the topic",
        backstory="Funny writer",
        llm=llm,
        verbose=True,
    )

    # Task 1
    task1 = Task(
        description="Pick a random animal.",
        expected_output="The name of an animal.",
        agent=researcher,
    )

    # Task 2
    task2 = Task(
        description="Write a one-sentence joke about the animal provided.",
        expected_output="A joke.",
        agent=writer,
        context=[task1],  # Explicit dependency
    )

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)

    return crew
