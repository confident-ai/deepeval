import os
import time
from crewai import Task, Crew
from deepeval.integrations.crewai.agent import Agent
from deepeval.integrations.crewai import instrument_crewai
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden

instrument_crewai(api_key="q8/AU3bxv2MX0mBnW9I8ynOVNx/iV3mMH3oqkl2Isu4=")

# Define your agents with roles and goals
coder = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
)

# Create tasks for your agents
task1 = Task(
    description="Explain the given topic",
    expected_output="A clear and concise explanation.",
    agent=coder,
)

# Instantiate your crew
crew = Crew(
    agents=[coder],
    tasks=[task1],
)

# # Kickoff your crew
# result = crew.kickoff(
#     inputs={"input": "What are the LLMs?"}
# )
# time.sleep(7) # Wait for traces to be posted to observatory

#################################

answer_relavancy_metric = AnswerRelevancyMetric()

coder = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    metrics=[answer_relavancy_metric],
)

goldens = [
    Golden(input="What are Transformers in AI?"),
    Golden(input="What is the biggest open source database?"),
    Golden(input="What are LLMs?"),
]

dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    result = crew.kickoff(inputs={"input": golden.input})

time.sleep(15)
