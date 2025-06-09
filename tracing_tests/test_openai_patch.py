from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.dataset import Golden
from deepeval.openai import openai, OpenAI, AsyncOpenAI
from deepeval import evaluate

client = OpenAI()

@observe(type="llm", model="gpt-4o", metrics=[AnswerRelevancyMetric()])
def your_llm_app(input: str, version: int = 1):
    if version == 1:
        response = client.responses.create(
            model="gpt-3.5-turbo",
            instructions="You are an assistant that talks like a pirate.",
            input=input,
            tools = [{
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
        )
        update_current_span(
            test_case=LLMTestCase(
                input=input,
                actual_output=response.output_text
            )
        )
        return response.output_text
    elif version == 2:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": input,
                },
            ],
            tools = [{
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
        )
        update_current_span(
            test_case=LLMTestCase(
                input=input,
                actual_output=response.choices[0].message.content
            )
        )
        return response.choices[0].message.content


# goldens = [
#     Golden(input="What is the weather like in Paris today?"),
#     Golden(input="What's the capital of Brazil?"),
#     # Golden(input="Who won the last World Cup?"),
#     # Golden(input="Explain quantum entanglement."),
#     # Golden(input="What's the latest iPhone model?"),
#     # Golden(input="How do I cook a perfect steak?"),
# ]
# test_cases = []
# for golden in goldens:
#     actual_output = your_llm_app(input=golden.input)
#     test_case = LLMTestCase(input=golden.input, actual_output=actual_output)
#     test_cases.append(test_case)

# evaluate(test_cases=test_cases, metrics=[AnswerRelevancyMetric()])


########################################################################
########################################################################
########################################################################

goldens = [
    Golden(input="What is the weather like in Paris today?"),
    Golden(input="What's the capital of Brazil?"),
    # Golden(input="Who won the last World Cup?"),
    # Golden(input="Explain quantum entanglement."),
    # Golden(input="What's the latest iPhone model?"),
    # Golden(input="How do I cook a perfect steak?"),
]
evaluate(observed_callback=your_llm_app, goldens=goldens)

########################################################################
########################################################################
########################################################################

your_llm_app("What is the weather like in Paris and Bangkok today?", 2)
