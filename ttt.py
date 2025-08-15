from deepeval.metrics import TaskCompletionMetric, ArgumentCorrectnessMetric

arg_correctness_metric = ArgumentCorrectnessMetric()
task_completion_metric = TaskCompletionMetric()

from openai import OpenAI
import json
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe, update_current_span

client = OpenAI()
tools = [...]


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
