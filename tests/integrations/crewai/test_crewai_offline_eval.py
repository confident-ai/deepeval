from crewai import Task, Crew
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.integrations.crewai.agent import Agent
import os
from deepeval.integrations.crewai import instrument_crewai
import time
from deepeval.dataset import Golden, EvaluationDataset

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
instrument_crewai(api_key=os.getenv("CONFIDENT_API_KEY"))

answer_relavancy_metric = AnswerRelevancyMetric()

coder = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    metrics=[answer_relavancy_metric],
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

goldens = [
    Golden(input="What are Transformers in AI?"),
    Golden(input="What is the biggest open source database?"),
    Golden(input="What are LLMs?"),
]

dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    result = crew.kickoff(inputs={"input": golden.input})
