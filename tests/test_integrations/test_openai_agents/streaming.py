import asyncio
from deepeval.openai_agents import Runner
from agents_app import weather_agent

runner = Runner()


async def main():
    result = runner.run_streamed(
        weather_agent,
        "What's the weather in UK?",
        metric_collection="test_collection_1",
    )
    async for chunk in result.stream_events():
        print(chunk, end="", flush=True)
        print("=" * 50)


# asyncio.run(main())
