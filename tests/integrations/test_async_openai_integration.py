from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.openai import OpenAI
from deepeval.tracing import observe

client = OpenAI()

##############################################
# Test end-to-end Evaluation
##############################################

def test_end_to_end_evaluation():
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


##############################################
# Test tracing
##############################################

@observe()
def llm_app(input: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": input},
        ],
        metrics=[AnswerRelevancyMetric(), BiasMetric()],
    )
    return response.choices[0].message.content

llm_app("hi")

##############################################
# Test tracing
##############################################

@observe()
def llm_app(input: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": input},
        ],
        metrics=[AnswerRelevancyMetric(), BiasMetric()],
    )
    return response.choices[0].message.content

llm_app("hi")