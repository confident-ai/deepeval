from deepeval.tracing import observe


@observe()
def your_ai_agent_tool():
    return "tool call result"


@observe()
def your_ai_agent(input):
    tool_call_result = your_ai_agent_tool()
    return "Tool Call Result: " + tool_call_result


your_ai_agent("Hello")
