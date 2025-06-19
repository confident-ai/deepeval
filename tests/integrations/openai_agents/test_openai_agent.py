from agents import set_default_openai_api, add_trace_processor
from deepeval.openai_agents import DeepEvalTracingProcessor
import asyncio
import shutil
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)
sys.path.insert(0, project_root)

# Import agents
from tests.integrations.openai_agents.streaming_guardrails_agent import (
    streaming_guardrails_agent,
)
from tests.integrations.openai_agents.output_guardrails_agent import (
    output_guardrails_agent,
)
from tests.integrations.openai_agents.code_interpreter_agent import (
    code_interpreter_agent,
)
from tests.integrations.openai_agents.customer_service_agent import (
    customer_service_agent,
)
from tests.integrations.openai_agents.streaming_agent import streaming_agent
from tests.integrations.openai_agents.research_agent import research_agent
from tests.integrations.openai_agents.remote_agent import remote_agent
from tests.integrations.openai_agents.thread_agent import thread_agent
from tests.integrations.openai_agents.git_mcp_agent import git_agent

add_trace_processor(DeepEvalTracingProcessor())

# Run agents
if __name__ == "__main__":
    if not shutil.which("uvx"):
        raise RuntimeError(
            "uvx is not installed. Please install it with `pip install uvx`."
        )

    # asyncio.run(git_agent())
    # asyncio.run(customer_service_agent())
    # asyncio.run(research_agent())
    # asyncio.run(code_interpreter_agent())
    # asyncio.run(remote_agent())
    # asyncio.run(streaming_agent())
    # asyncio.run(streaming_guardrails_agent())
    # asyncio.run(output_guardrails_agent())
    asyncio.run(thread_agent())

    ##########################################
    # Test Async
    ##########################################

    async def gather_research_agents():
        tasks = [
            research_agent(query="What is the stock price of Apple?")
            for _ in range(10)
        ]
        await asyncio.gather(*tasks)

    # asyncio.run(gather_research_agents())

    ##########################################
    # Test Async Streaming
    ##########################################

    async def gather_streaming_agents():
        tasks = [streaming_agent() for _ in range(4)]
        await asyncio.gather(*tasks)

    # asyncio.run(gather_streaming_agents())
