from deepeval.anthropic import Anthropic
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.tracing import observe

client = Anthropic()

@observe(
    type="llm",
    model="claude-sonnet-4-5"
)
def generate_response(input: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": input,
            }
        ],
    )
    return response

response = generate_response("Hey, how are you?")

##############################################

@observe(
    type="llm",
    model="claude-sonnet-4-5"
)
def generate_response(input: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello, Claude"
            }
        ],
        metrics=[AnswerRelevancyMetric()],
    )
    return response

goldens = [
    Golden(input="What is application of useState() in React?"),
    Golden(input="Compare Repeatable Reads vs Read Committed as Isolation level for PostgreSQL."),
]


dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    result = generate_response(golden.input)
    print(f"Input: {golden.input}\nResponse: {result}\n{'-'*50}")
