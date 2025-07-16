from deepeval.tracing import (
    observe,
    update_current_span,
    LlmAttributes,
)
import asyncio


@observe(
    type="llm",
    model="gemini-2.5-flash",
    cost_per_input_token=0.0000003,
    cost_per_output_token=0.0000025,
)
async def meta_agent(query: str):
    update_current_span(
        attributes=LlmAttributes(
            input=query,
            output=query,
            input_token_count=10,
            output_token_count=10,
        )
    )
    return query


async def run_parallel_examples():
    tasks = [
        meta_agent("How tall is Mount Everest?"),
        meta_agent("What's the capital of Brazil?"),
        # meta_agent("Who won the last World Cup?"),
        # meta_agent("Explain quantum entanglement."),
        # meta_agent("What's the latest iPhone model?"),
        # meta_agent("How do I cook a perfect steak?"),
        # meta_agent("Tell me a joke about robots."),
        # meta_agent("What causes lightning?"),
        # meta_agent("Who painted the Mona Lisa?"),
        # meta_agent("What's the population of Japan?"),
        # meta_agent("How do vaccines work?"),
        # meta_agent("Recommend a good sci-fi movie."),
    ]
    await asyncio.gather(*tasks)


asyncio.run(run_parallel_examples())
