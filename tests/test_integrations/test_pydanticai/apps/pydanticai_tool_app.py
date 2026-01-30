"""
PydanticAI Tool App: Agent with tool calling
Complexity: MEDIUM - Tests tool calling functionality

Uses deterministic settings (temperature=0) for reproducible traces.
"""

from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai import ConfidentInstrumentationSettings


def create_tool_agent(
    name: str = "pydanticai-tool-test",
    tags: list = None,
    metadata: dict = None,
    thread_id: str = None,
    user_id: str = None,
    tool_metric_collection_map: dict = None,
) -> Agent:
    """Create a PydanticAI agent with tools and instrumentation settings."""
    settings = ConfidentInstrumentationSettings(
        name=name,
        tags=tags or ["pydanticai", "tool"],
        metadata=metadata or {"test_type": "tool"},
        thread_id=thread_id,
        user_id=user_id,
        tool_metric_collection_map=tool_metric_collection_map or {},
        is_test_mode=True,
    )

    agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt=(
            "You are a calculator assistant. Use the calculate tool "
            "for math operations. Be concise."
        ),
        instrument=settings,
        name="tool_agent",
    )

    @agent.tool_plain
    def calculate(operation: str, a: float, b: float) -> float:
        """
        Perform basic arithmetic operations.

        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number

        Returns:
            The result of the operation
        """
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

    return agent


def invoke_tool_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the tool agent synchronously."""
    if agent is None:
        agent = create_tool_agent()
    result = agent.run_sync(prompt)
    return result.output


async def ainvoke_tool_agent(prompt: str, agent: Agent = None) -> str:
    """Invoke the tool agent asynchronously."""
    if agent is None:
        agent = create_tool_agent()
    result = await agent.run(prompt)
    return result.output
