from openai import OpenAI
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import ArgumentCorrectnessMetric, TaskCompletionMetric
import json

...

arg_correctness_metric = ArgumentCorrectnessMetric()
task_completion_metric = TaskCompletionMetric()
client = OpenAI()
tools = [
    {
        "type": "function",
        "name": "web_search_tool",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {"web_query": {"type": "string"}},
            "required": ["web_query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


@observe()
def web_search_tool(web_query):
    return "Web search results"


# Supply metric
@observe(metrics=[arg_correctness_metric])
def llm_component(query):
    response = client.responses.create(
        model="gpt-4.1", input=[{"role": "user", "content": query}], tools=tools
    )

    # Format tools
    tools_called = [
        ToolCall(name=tool_call.name, arguments=tool_call.arguments)
        for tool_call in response.output
        if tool_call.type == "function_call"
    ]

    # Create test cases on the component-level
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response.output_text,
            tools_called=tools_called,
        )
    )
    return response.output


# Supply metric
@observe(metrics=[task_completion_metric])
def your_ai_agent(query: str) -> str:
    llm_output = llm_component(query)
    search_results = "".join(
        [
            web_search_tool(**json.loads(tool_call.arguments))
            for tool_call in llm_output
            if tool_call == "function_call"
        ]
    )
    return "The answer to your question is: " + search_results


your_ai_agent("What are LLMs?")
