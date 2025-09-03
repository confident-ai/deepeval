import os
import time
from crewai import Task, Crew, Agent
from crewai.tools import tool

from deepeval.integrations.crewai import instrument_crewai

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

instrument_crewai(api_key=os.getenv("CONFIDENT_API_KEY"))


@tool
def get_weather(city: str):
    """Get the weather"""
    return f"The weather is of {city} is sunny"


# Define your agents with roles and goals
coder = Agent(
    role="Weather Agent",
    goal="Get the weather",
    backstory="An agent that gets the weather",
    tools=[get_weather],
    metric_collection="Task Completion Collection",
)

# Create tasks for your agents
task1 = Task(
    description="Get the weather",
    expected_output="The weather is of San Francisco",
    agent=coder,
)

# Instantiate your crew
crew = Crew(agents=[coder], tasks=[task1], memory=True)

# Kickoff your crew
result = crew.kickoff()
print(result)

time.sleep(7)  # Wait for traces to be posted to observatory
