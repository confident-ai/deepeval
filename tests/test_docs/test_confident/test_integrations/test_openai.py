from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.openai import OpenAI
from deepeval.tracing import observe

client = OpenAI()


@observe(type="llm", model="gpt-4.1")
def generate_response(input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ],
    )
    return response


response = generate_response("What is the weather in Tokyo?")

############################################


client = OpenAI()


@observe(type="llm", model="gpt-4.1")
def generate_response(input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ],
        metrics=[AnswerRelevancyMetric()],
    )
    return response


# Create goldens
goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
dataset = EvaluationDataset(goldens=goldens)

# Run component-level evaluation
for golden in dataset.evals_iterator():
    generate_response(golden.input)
