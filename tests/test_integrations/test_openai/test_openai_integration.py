from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.evaluate import dataset
from deepeval.dataset import Golden
from deepeval.openai import OpenAI

from tests.integrations.openai.resources import (
    llm_app,
    CHAT_TOOLS,
    RESPONSE_TOOLS,
)

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]


def test_end_to_end_loop():
    openai_client = OpenAI()
    for golden in dataset(goldens=goldens):
        openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful chatbot. Always generate a string response.",
                },
                {"role": "user", "content": golden.input},
            ],
            tools=CHAT_TOOLS,
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        )
    for golden in dataset(goldens=goldens):
        openai_client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful chatbot. Always generate a string response.",
            input=golden.input,
            tools=RESPONSE_TOOLS,
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        )


def test_component_level_loop():
    for golden in dataset(goldens=goldens):
        llm_app(golden.input, completion_mode="chat")

    for golden in dataset(goldens=goldens):
        llm_app(golden.input, completion_mode="response")


def test_tracing():
    llm_app("What is the weather in Bogotá, Colombia?", completion_mode="chat")
    llm_app("What is the weather in Paris, France?", completion_mode="chat")
    llm_app(
        "What is the weather in Bogotá, Colombia?", completion_mode="response"
    )
    llm_app("What is the weather in Paris, France?", completion_mode="response")


##############################################
# Test Everything
##############################################

if __name__ == "__main__":
    # test_end_to_end_loop()
    # test_component_level_loop()
    test_tracing()
