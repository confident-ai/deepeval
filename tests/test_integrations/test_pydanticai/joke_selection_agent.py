from dataclasses import dataclass
import asyncio
import httpx

from pydantic_ai import Agent, RunContext
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)


@dataclass
class ClientAndKey:
    http_client: httpx.AsyncClient
    api_key: str


joke_selection_agent = Agent(
    "openai:gpt-4o",
    deps_type=ClientAndKey,
    system_prompt=(
        "Use the `joke_factory` tool to generate some jokes on the given subject, "
        "then choose the best. You must return just a single joke."
    ),
    instrument=ConfidentInstrumentationSettings(
        thread_id="test_thread_id_1",
        agent_metric_collection="test_collection_1",
        llm_metric_collection="test_collection_1",
    ),
)
joke_generation_agent = Agent(
    "openai:gpt-4o",
    deps_type=ClientAndKey,
    output_type=list[str],
    system_prompt=(
        'Use the "get_jokes" tool to get some jokes on the given subject, '
        "then extract each joke into a list."
    ),
    instrument=ConfidentInstrumentationSettings(),
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
        "https://example.com",
        params={"count": count},
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
    )
    response.raise_for_status()
    return response.text


async def _execute_joke_selection_agent():
    async with httpx.AsyncClient() as client:
        deps = ClientAndKey(client, "foobar")
        result = await joke_selection_agent.run("Tell me a joke.", deps=deps)
        print("===============Joke selection agent output:===============")
        print(result.output)


def execute_joke_selection_agent():
    asyncio.run(_execute_joke_selection_agent())
