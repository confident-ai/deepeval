from crewai import Task, Crew
from deepeval.integrations.crewai import Agent
import os
from deepeval.integrations.crewai import instrument_crewai
import time


# Define your agents with roles and goals
coder = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    metric_collection="Task Completion Collection",
)

# Create tasks for your agents
task1 = Task(
    description="Explain the latest trends in AI.",
    expected_output="A clear and concise explanation.",
    agent=coder,
)

# Instantiate your crew
crew = Crew(
    agents=[coder],
    tasks=[task1],
)

# Kickoff your crew
result = crew.kickoff()
