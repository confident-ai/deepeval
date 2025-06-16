from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.openai import OpenAI

client = OpenAI()

for i in range(5):
    client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        metrics=[AnswerRelevancyMetric()],
    )

for i in range(5):
    client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": "Hello!"},
        ],
        metrics=[AnswerRelevancyMetric(), BiasMetric()],
    )
