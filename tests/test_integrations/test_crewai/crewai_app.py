# this is not working
from crewai import Task, Crew, Agent
from deepeval.integrations.crewai import instrument_crewai

instrument_crewai()

agent = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
)

task = Task(
    description="Explain the given topic",
    expected_output="A clear and concise explanation.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
)

result = crew.kickoff({"input": "What are the LLMs?"})

print(result)