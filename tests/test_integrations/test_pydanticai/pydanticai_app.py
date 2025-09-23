from __future__ import annotations as _annotations
import asyncio
from dataclasses import dataclass
from typing import Any

from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import RunContext
from deepeval.integrations.pydantic_ai import Agent
from deepeval.prompt import Prompt

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")


@dataclass
class Deps:
    client: AsyncClient


weather_agent = Agent(
    "openai:gpt-4o-mini",
    llm_metric_collection="test_collection_1",
    llm_prompt=prompt,
    agent_metric_collection="test_collection_1",
    instructions="Be concise, reply with one sentence.",
    deps_type=Deps,
    retries=2,
    trace_name="test_trace_1",
    trace_tags=["test_tag_1"],
    trace_metadata={"test_metadata_1": "test_metadata_1"},
    trace_thread_id="test_thread_id_1",
    trace_user_id="test_user_id_1",
    trace_metric_collection="test_collection_1",
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool(metric_collection="test_collection_1")
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> LatLng:

    r = await ctx.deps.client.get(
        "https://demo-endpoints.pydantic.workers.dev/latlng",
        params={"location": location_description},
    )
    r.raise_for_status()
    return LatLng.model_validate_json(r.content)


@weather_agent.tool(metric_collection="test_collection_1")
async def get_weather(
    ctx: RunContext[Deps], lat: float, lng: float
) -> dict[str, Any]:

    temp_response, descr_response = await asyncio.gather(
        ctx.deps.client.get(
            "https://demo-endpoints.pydantic.workers.dev/number",
            params={"min": 10, "max": 30},
        ),
        ctx.deps.client.get(
            "https://demo-endpoints.pydantic.workers.dev/weather",
            params={"lat": lat, "lng": lng},
        ),
    )
    temp_response.raise_for_status()
    descr_response.raise_for_status()
    return {
        "temperature": f"{temp_response.text} °C",
        "description": descr_response.text,
    }


async def run_agent(input_query: str):
    async with AsyncClient() as client:
        deps = Deps(client=client)
        result = await weather_agent.run(
            input_query,
            deps=deps,
            # metric_collection="test_collection_1",
            name="test_trace_2",
            # tags=["test_tag_1"],
            # metadata={"test_metadata_1": "test_metadata_1"},
            # thread_id="test_thread_id_1",
            user_id="test_user_id_2",
        )

        return result.output


def execute_agent():
    output = asyncio.run(run_agent("What's the weather in Paris?"))
    return output
