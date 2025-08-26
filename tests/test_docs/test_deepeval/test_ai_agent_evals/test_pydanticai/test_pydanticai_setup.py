import time
from pydantic_ai import Agent

from deepeval.integrations.pydantic_ai import instrument_pydantic_ai
instrument_pydantic_ai(api_key="<your-confident-api-key>")

agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="Be concise, reply with one sentence.",
)

result = agent.run_sync("What are the LLMs?")
print(result)
time.sleep(10) # wait for the trace to be posted

# running agent in async mode
# import asyncio
# async def main():
#     result = await agent.run("What are the LLMs?")
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(main())
#     time.sleep(10)