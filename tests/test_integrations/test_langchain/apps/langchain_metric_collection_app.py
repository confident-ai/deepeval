"""
Metric Collection LangChain App: Tests metric_collection on LLM and tool spans
Complexity: LOW - Tests metric_collection tracing

Uses ChatOpenAI with metric_collection in metadata and the patched @tool decorator
with metric_collection for component-level evaluations.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from deepeval.integrations.langchain import tool
from deepeval.prompt import Prompt

# Create a Prompt object for prompt tracking
test_prompt = Prompt(alias="metric-collection-test-prompt")
test_prompt.version = "01.00.00"


@tool(metric_collection="tool_accuracy")
def calculate(expression: str) -> str:
    """Evaluates a simple math expression and returns the result."""
    # Simple calculator that handles basic operations
    try:
        # Only allow safe characters
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# LLM with metric_collection and prompt in metadata
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=42,
    metadata={
        "metric_collection": "llm_quality",
        "prompt": test_prompt,
    },
)

# Create agent with the tool
agent_executor = create_agent(
    llm,
    [calculate],
    system_prompt="You are a calculator assistant. Use the calculate tool to evaluate math expressions.",
)


def invoke_metric_collection_app(inputs: dict, config: RunnableConfig = None):
    """Invoke the metric collection app."""
    return agent_executor.invoke(inputs, config=config)


async def ainvoke_metric_collection_app(
    inputs: dict, config: RunnableConfig = None
):
    """Async invoke the metric collection app."""
    return await agent_executor.ainvoke(inputs, config=config)
