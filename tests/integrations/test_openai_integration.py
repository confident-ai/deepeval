from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.tracing import observe
from deepeval.dataset import Golden
from deepeval.openai import OpenAI
from deepeval import evaluate

COMPLETION_TOOLS = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}] 

CHAT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

completion_mode = "chat"
client = OpenAI()
tools = CHAT_TOOLS if completion_mode == "chat" else COMPLETION_TOOLS

##############################################
# Test end-to-end Evaluation
##############################################

def test_end_to_end_evaluation():
    if completion_mode == "chat":
        for i in range(5):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always generate a string response."},
                    {"role": "user", "content": "What is the weather in Bogotá, Colombia?"},
                ],
                tools=tools,
                metrics=[AnswerRelevancyMetric()],
            )
    else:
        for i in range(5):
            client.responses.create(
                model="gpt-4o",
                instructions="You are a helpful assistant. Always generate a string response.",
                input="What is the weather in Bogotá, Colombia?",
                tools=tools,
                metrics=[AnswerRelevancyMetric()],
            )

##############################################
# Test tracing
##############################################

@observe()
def llm_app(input: str):
    if completion_mode == "chat":
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot. Always generate a string response."},
                {"role": "user", "content": input},
            ],
            tools=tools,
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        )
        return response.choices[0].message.content
    else:
        response = client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant. Always generate a string response.",
            input=input,
            tools=tools,
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        )
        return response.output_text

def test_tracing():
    llm_app("What is the weather in Bogotá, Colombia?")
    llm_app("What is the weather in Paris, France?")

##############################################
# Test traceable evaluate
##############################################

def test_traceable_evaluate():
    evaluate(
        observed_callback=llm_app,
        goldens=[
            Golden(input="What is the weather in Bogotá, Colombia?"),
            Golden(input="What is the weather in Paris, France?"),
        ],
    )

##############################################
# Test Everything
##############################################

if __name__ == "__main__":
    test_end_to_end_evaluation()
    test_traceable_evaluate()
    test_tracing()