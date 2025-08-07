from dataclasses import dataclass
import os
import time
import httpx
import asyncio
from deepeval.integrations.pydantic_ai import Agent
from pydantic_ai import RunContext

from dotenv import load_dotenv


load_dotenv()
from deepeval.integrations.pydantic_ai import instrument_pydantic_ai

instrument_pydantic_ai(api_key=os.getenv("CONFIDENT_API_KEY"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
Agent.instrument_all()


@dataclass
class ClientAndKey:
    http_client: httpx.AsyncClient
    api_key: str


joke_selection_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ClientAndKey,
    system_prompt=(
        "Use the `joke_factory` tool to generate some jokes on the given subject, "
        "then choose the best. You must return just a single joke."
    ),
)
joke_generation_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ClientAndKey,
    output_type=list[str],
    system_prompt=(
        'Use the "get_jokes" tool to get some jokes on the given subject, '
        "then extract each joke into a list."
    ),
)


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[ClientAndKey], count: int) -> list[str]:
    r = await joke_generation_agent.run(
        f"Please generate {count} jokes.",
        deps=ctx.deps,
        usage=ctx.usage,
    )
    return r.output


@joke_generation_agent.tool
async def get_jokes(ctx: RunContext[ClientAndKey], count: int) -> str:
    response = await ctx.deps.http_client.get(
        "https://jsonplaceholder.typicode.com/posts",
        params={"count": count},
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
    )
    response.raise_for_status()
    return response.text


async def main():
    async with httpx.AsyncClient() as client:
        deps = ClientAndKey(client, "foobar")
        result = await joke_selection_agent.run("Tell me a joke.", deps=deps)
        print(result.output)
        # > Did you hear about the toothpaste scandal? They called it Colgate.
        print(result.usage())
        # > Usage(requests=4, request_tokens=309, response_tokens=32, total_tokens=341)
        # > Did you hear about the toothpaste scandal? They called it Colgate.
        print(result.usage())
        # > Usage(requests=4, request_tokens=309, response_tokens=32, total_tokens=341)


if __name__ == "__main__":
    asyncio.run(main())
    time.sleep(10)
