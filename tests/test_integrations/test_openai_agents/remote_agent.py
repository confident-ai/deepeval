import asyncio
from deepeval.openai_agents import Runner, Agent, DeepEvalTracingProcessor
from agents import add_trace_processor

add_trace_processor(DeepEvalTracingProcessor())

URL = "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"


async def remote_agent():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
    )

    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "detail": "auto", "image_url": URL}
                ],
            },
            {
                "role": "user",
                "content": "What do you see in this image?",
            },
        ],
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(remote_agent())