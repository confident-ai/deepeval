from deepeval.openai import OpenAI
from deepeval.metrics import AnswerRelevancyMetric

client = OpenAI()

client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    metrics=[AnswerRelevancyMetric()]
)

client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hiihi"}
    ],
    metrics=[AnswerRelevancyMetric()]
)