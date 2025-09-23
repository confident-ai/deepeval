import asyncio
from deepeval.openai_agents import Agent, Runner
from deepeval.prompt import Prompt
from deepeval.openai_agents import DeepEvalTracingProcessor

from agents import add_trace_processor

add_trace_processor(DeepEvalTracingProcessor())

prompt = Prompt(alias="asd")
prompt.pull(version="00.00.01")

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    confident_prompt=prompt,
    llm_metric_collection="test_collection_1",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    # result = await Runner.run(
    #     triage_agent, input="Hola, ¿cómo estás?",
    #     metric_collection="test_collection_1",
    #     tags=["test"],
    #     thread_id="test",
    # )

    runner = Runner()
    result = runner.run_streamed(
        triage_agent,
        "Hola, ¿cómo estás?",
        metric_collection="test_collection_1",
        thread_id="test",
    )
    async for chunk in result.stream_events():
        print(chunk, end="", flush=True)
        print("=" * 50)


# asyncio.run(main())
