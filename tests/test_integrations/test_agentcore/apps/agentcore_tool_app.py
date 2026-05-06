from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool

from deepeval.integrations.agentcore import instrument_agentcore


@tool
def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    op_func = operations.get(operation.lower())
    if op_func is None:
        raise ValueError(f"Unsupported operation: {operation}")
    return op_func(a, b)


def init_tool_agentcore(
    name: str = "agentcore-tool-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    """Trace-only setup. Tool / agent / LLM span-level fields belong at
    the call site (``with next_*_span(...)`` or ``update_current_span``
    inside the tool body)."""
    instrument_agentcore(
        name=name,
        tags=tags or ["agentcore", "tool"],
        metadata=metadata or {"test_type": "tool"},
        thread_id=thread_id,
        user_id=user_id,
    )

    app = BedrockAgentCoreApp()
    agent = Agent(model="amazon.nova-lite-v1:0", tools=[calculate])

    @app.entrypoint
    def invoke(payload: dict):
        user_message = payload.get("prompt", "What is 7 multiplied by 8?")
        instruction = "You are a calculator assistant. Use the calculate tool for math operations. Be concise. "
        result = agent(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    async def ainvoke(payload: dict):
        user_message = payload.get("prompt", "What is 7 multiplied by 8?")
        instruction = "You are a calculator assistant. Use the calculate tool for math operations. Be concise. "
        result = await agent.invoke_async(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    invoke.ainvoke = ainvoke
    return invoke


def invoke_tool_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_tool_agentcore()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_tool_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_tool_agentcore()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
