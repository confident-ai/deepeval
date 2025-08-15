from crewai import Task, Crew

from deepeval.integrations.crewai import Agent
from deepeval.integrations.crewai import instrument_crewai
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden

instrument_crewai()

answer_relavancy_metric = AnswerRelevancyMetric()

agent = Agent(
    role="Consultant",
    goal="Write clear, concise explanation.",
    backstory="An expert consultant with a keen eye for software trends.",
    metrics=[answer_relavancy_metric],
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

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What are Transformers in AI?"),
        Golden(input="What is the biggest open source database?"),
        Golden(input="What are LLMs?"),
    ]
)

for golden in dataset.evals_iterator():
    result = crew.kickoff(inputs={"input": golden.input})
