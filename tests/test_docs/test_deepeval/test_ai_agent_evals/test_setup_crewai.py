from deepeval.integrations.crewai import instrumentator, Agent
from crewai import Task, Crew

instrumentator(api_key="q8/AU3bxv2MX0mBnW9I8ynOVNx/iV3mMH3oqkl2Isu4=")

coder = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
)

task = Task(
    description="Explain the latest trends in AI.",
    agent=coder,
    expected_output="A clear and concise explanation.",
)

crew = Crew(
    agents=[coder],
    tasks=[task],
)
result = crew.kickoff()
