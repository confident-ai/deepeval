from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import RunContext

from deepeval.integrations.pydantic_ai import instrument_pydantic_ai, Agent

instrument_pydantic_ai(api_key=os.getenv("CONFIDENT_API_KEY"))


@dataclass
class Deps:
    client: AsyncClient


weather_agent = Agent(
    "openai:gpt-4o-mini",
    instructions="Be concise, reply with one sentence.",
    deps_type=Deps,
    retries=2,
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool(metric_collection="test_collection_1")
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> LatLng:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    # current_span = trace.get_current_span()

    # # You can now interact with the span, for example, by adding attributes
    # if current_span.is_recording():
    #     current_span.set_attribute("confident.span.output", "Hello")

    # NOTE: the response here will be random, and is not related to the location description.

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
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    # NOTE: the responses here will be random, and are not related to the lat and lng.
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
        result = await weather_agent.run(input_query, deps=deps)
        return result.output


# run the agent with a sample input and print the result
import asyncio
import time

if __name__ == "__main__":
    input_query = "What's the weather in Paris?"
    output = asyncio.run(run_agent(input_query))
    print("Agent output:", output)
    time.sleep(10)
