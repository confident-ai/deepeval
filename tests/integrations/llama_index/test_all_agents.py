from .test_llama_index_codeact_agent import run_agent_verbose, agent as codeact_agent, ctx as codeact_ctx
from .test_llama_index_react_agent import agent as react_agent
from .test_llama_index_function_agent import agent as function_agent
import asyncio
import time

async def main():
    await run_agent_verbose(codeact_agent, codeact_ctx, "Calculate the sum of all numbers from 1 to 10")
    await react_agent.run("What is the capital of France?")
    await function_agent.run("What's 7 * 8?")

if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(7)