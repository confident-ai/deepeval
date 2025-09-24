import asyncio
from agents import add_trace_processor
from deepeval.openai_agents.callback_handler import DeepEvalTracingProcessor

add_trace_processor(DeepEvalTracingProcessor())

from agents_app import execute_agent
from run_sync import run_sync
from run_streamed import run_streamed
from run import run

async def run_main():
    await run()
    await run_streamed()
    await execute_agent()

def execute_all():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_main())
        run_sync()  # Now there's an active event loop
    finally:
        loop.close()

execute_all()