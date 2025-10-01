import asyncio
from weather_agent import weather_agent
from agents import Runner, add_trace_processor
from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor

add_trace_processor(DeepEvalTracingProcessor())


async def run():
    await Runner.run(
        weather_agent,
        "What's the weather in London?",
    )


# asyncio.run(run())
