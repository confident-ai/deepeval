import os

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from deepeval.integrations.strands import instrument_strands


_DEFAULT_MODEL_ID = os.environ.get("STRANDS_TEST_MODEL", "gpt-4o-mini")


def _build_openai_model() -> OpenAIModel:
    return OpenAIModel(
        client_args={"api_key": os.environ.get("OPENAI_API_KEY", "")},
        model_id=_DEFAULT_MODEL_ID,
        params={"temperature": 0.0},
    )


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


def init_tool_strands(
    name: str = "strands-tool-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
):
    """Trace-only setup. Tool / agent / LLM span-level fields belong at
    the call site (``with next_*_span(...)`` or ``update_current_span``
    inside the tool body)."""
    instrument_strands(
        name=name,
        tags=tags or ["strands", "tool"],
        metadata=metadata or {"test_type": "tool"},
        thread_id=thread_id,
        user_id=user_id,
    )

    agent = Agent(model=_build_openai_model(), tools=[calculate])

    def invoke(payload: dict):
        user_message = payload.get("prompt", "What is 7 multiplied by 8?")
        instruction = (
            "You are a calculator assistant. "
            "Use the calculate tool for math operations. Be concise. "
        )
        result = agent(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    async def ainvoke(payload: dict):
        user_message = payload.get("prompt", "What is 7 multiplied by 8?")
        instruction = (
            "You are a calculator assistant. "
            "Use the calculate tool for math operations. Be concise. "
        )
        result = await agent.invoke_async(instruction + user_message)

        text_output = result.message.get("content", [{}])[0].get("text", "")
        return {"result": text_output}

    invoke.ainvoke = ainvoke
    return invoke


def invoke_tool_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_tool_strands()
    response = invoke_func({"prompt": prompt})
    return response.get("result", "")


async def ainvoke_tool_agent(prompt: str, invoke_func=None) -> str:
    if invoke_func is None:
        invoke_func = init_tool_strands()
    response = await invoke_func.ainvoke({"prompt": prompt})
    return response.get("result", "")
