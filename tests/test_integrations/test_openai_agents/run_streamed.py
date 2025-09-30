import asyncio
from weather_agent import weather_agent
from agents import Runner, add_trace_processor
from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor

add_trace_processor(DeepEvalTracingProcessor())


async def run_streamed():
    run_streamed = Runner.run_streamed(
        weather_agent,
        "What's the weather in London?",
    )

    async for chunk in run_streamed.stream_events():
        continue


if __name__ == "__main__":
    asyncio.run(run_streamed())
