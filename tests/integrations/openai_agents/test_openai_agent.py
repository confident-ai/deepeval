from deepeval.openai_agents import trace_openai_agents
import asyncio
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from tests.integrations.openai_agents.customer_service import customer_service_agent

trace_openai_agents()

if __name__ == "__main__":
    asyncio.run(customer_service_agent())