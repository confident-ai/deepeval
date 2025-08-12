import os
from deepeval.integrations.llama_index import instrument_llama_index
import llama_index.core.instrumentation as instrument
from llama_index.core.workflow import Context
from deepeval.integrations.llama_index import CodeActAgent
import asyncio
import time

from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import deepeval

deepeval.login(os.getenv("CONFIDENT_API_KEY"))
instrument_llama_index(instrument.get_dispatcher())


# Define a few helper functions
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b


from typing import Any, Dict, Tuple
import io
import contextlib
import ast
import traceback


class SimpleCodeExecutor:
    """
    A simple code executor that runs Python code with state persistence.

    This executor maintains a global and local state between executions,
    allowing for variables to persist across multiple code runs.

    NOTE: not safe for production use! Use with caution.
    """

    def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
        """
        Initialize the code executor.

        Args:
            locals: Local variables to use in the execution context
            globals: Global variables to use in the execution context
        """
        # State that persists between executions
        self.globals = globals
        self.locals = locals

    def execute(self, code: str) -> Tuple[bool, str, Any]:
        """
        Execute Python code and capture output and return values.

        Args:
            code: Python code to execute

        Returns:
            Dict with keys `success`, `output`, and `return_value`
        """
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()

        output = ""
        return_value = None
        try:
            # Execute with captured output
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                # Try to detect if there's a return value (last expression)
                try:
                    tree = ast.parse(code)
                    last_node = tree.body[-1] if tree.body else None

                    # If the last statement is an expression, capture its value
                    if isinstance(last_node, ast.Expr):
                        # Split code to add a return value assignment
                        last_line = code.rstrip().split("\n")[-1]
                        exec_code = (
                            code[: -len(last_line)]
                            + "\n__result__ = "
                            + last_line
                        )

                        # Execute modified code
                        exec(exec_code, self.globals, self.locals)
                        return_value = self.locals.get("__result__")
                    else:
                        # Normal execution
                        exec(code, self.globals, self.locals)
                except:
                    # If parsing fails, just execute the code as is
                    exec(code, self.globals, self.locals)

            # Get output
            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()

        except Exception as e:
            # Capture exception information
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()

        if return_value is not None:
            output += "\n\n" + str(return_value)

        return output


code_executor = SimpleCodeExecutor(
    # give access to our functions defined above
    locals={
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    },
    globals={
        # give access to all builtins
        "__builtins__": __builtins__,
        # give access to numpy
        "np": __import__("numpy"),
    },
)

agent = CodeActAgent(
    code_execute_fn=code_executor.execute,
    llm=OpenAI(model="gpt-4o-mini"),
    tools=[add, subtract, multiply, divide],
)

# context to hold the agent's session/state/chat history
ctx = Context(agent)

agent = CodeActAgent(
    code_execute_fn=code_executor.execute,
    llm=OpenAI(model="gpt-4o-mini"),
    tools=[add, subtract, multiply, divide],
    metric_collection="test_collection_1",
)

# context to hold the agent's session/state/chat history
ctx = Context(agent)

from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
    AgentStream,
)


async def run_agent_verbose(agent, ctx, query):
    handler = agent.run(query, ctx=ctx)
    print(f"User:  {query}")
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            print(f"\n-----------\nCode execution result:\n{event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"\n-----------\nParsed code:\n{event.tool_kwargs['code']}")
        elif isinstance(event, AgentStream):
            print(f"{event.delta}", end="", flush=True)

    return await handler


async def main():
    response = await run_agent_verbose(
        agent, ctx, "Calculate the sum of all numbers from 1 to 10"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(7)
