from deepeval.openai_agents import trace_openai_agents
import asyncio
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from tests.integrations.openai_agents.customer_service import customer_service_agent
from tests.integrations.openai_agents.research_agent import FinancialResearchManager

trace_openai_agents()

async def main() -> None:
    query = input("Enter a financial research query: ")
    mgr = FinancialResearchManager()
    await mgr.run(query)

if __name__ == "__main__":
    # asyncio.run(customer_service_agent())
    asyncio.run(main())