from pydantic_ai import Agent
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)

# Set your OpenAI API key
# os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Create an agent
agent = Agent(
    "openai:gpt-4o-mini",  # You can use any supported model
    system_prompt=(
        "You are a helpful calculator assistant. "
        "Use the calculate tool to perform arithmetic operations. "
        "Always show your work and explain the result."
    ),
    instrument=ConfidentInstrumentationSettings(),
)


# Define a simple tool using @agent.tool_plain (no context needed)
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
        "divide": lambda x, y: x / y if y != 0 else None,
    }

    op_func = operations.get(operation.lower())
    if op_func is None:
        raise ValueError(f"Unsupported operation: {operation}")

    result = op_func(a, b)
    if result is None:
        raise ValueError("Cannot divide by zero")

    return result


@agent.tool_plain
def power(base: float, exponent: float) -> float:
    """
    Calculate power and roots.

    Args:
        base: The base number
        exponent: The exponent (use 0.5 for square root, 0.333 for cube root, etc.)

    Returns:
        The result of base raised to the exponent
    """
    try:
        result = base**exponent
        return result
    except Exception as e:
        raise ValueError(f"Cannot calculate {base}^{exponent}: {str(e)}")


# Run the agent
if __name__ == "__main__":
    # Example 1: Simple calculation
    result = agent.run_sync("What is 15 multiplied by 7?")
    print("Result:", result.output)
    print("\n" + "=" * 50 + "\n")

    # Example 2: More complex query
    result = agent.run_sync("If I have 100 and divide it by 4, what do I get?")
    print("Result:", result.output)
    print("\n" + "=" * 50 + "\n")

    # View the messages exchanged
    print("Messages exchanged:")
    for message in result.all_messages():
        print(f"\n{message}")
