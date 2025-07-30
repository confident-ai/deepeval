from crewai import Task, Crew
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.integrations.crewai.agent import Agent
import os
from deepeval.integrations.crewai import instrumentator
import time
from deepeval.dataset import Golden
from deepeval.evaluate import dataset

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"
instrumentator(api_key="<YOUR_CONFIDENT_API_KEY>")

answer_relavancy_metric = AnswerRelevancyMetric()

# Define your agents with roles and goals
coder = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    # metric_collection="test_collection_1",
    metrics=[answer_relavancy_metric],
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

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
    Golden(input="What is the weather in Tokyo, Japan?"),
]

for golden in dataset(goldens=goldens):
    # Kickoff your crew
    result = crew.kickoff()

# time.sleep(7) # Wait for traces to be posted to observatory
