from __future__ import annotations as _annotations
import asyncio
from dataclasses import dataclass
from typing import Any

from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import RunContext, Agent
from deepeval.prompt import Prompt
from deepeval.integrations.pydantic_ai.instrumentator import (
    ConfidentInstrumentationSettings,
)
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import Golden, EvaluationDataset

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")


@dataclass
class Deps:
    client: AsyncClient


weather_agent = Agent(
    "openai:gpt-4o-mini",
    instructions="Be concise, reply with one sentence.",
    deps_type=Deps,
    retries=2,
    instrument=ConfidentInstrumentationSettings(
        is_test_mode=True, agent_metrics=[AnswerRelevancyMetric()]
    ),
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool()
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> LatLng:

    r = await ctx.deps.client.get(
        "https://demo-endpoints.pydantic.workers.dev/latlng",
        params={"location": location_description},
    )
    r.raise_for_status()
    return LatLng.model_validate_json(r.content)


@weather_agent.tool()
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
        )
        print(result.output)

        return result.output


dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's the weather in Paris?"),
        Golden(input="What's the weather in London?"),
    ]
)

if __name__ == "__main__":
    for golden in dataset.evals_iterator():
        task = asyncio.create_task(run_agent(golden.input))
        dataset.evaluate(task)
