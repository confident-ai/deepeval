from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import (
    observe,
    update_current_span,
    update_current_trace,
)
import asyncio


@observe(
    type="agent",
    agent_handoffs=["weather_agent", "research_agent", "custom_research_agent"],
    # metrics=[
    #     "Answer Relevancy",
    #     "Faithfulness",
    #     # "Helpfulness",
    #     # "Verbosity",
    #     # "Contextual Precision",
    #     # "Contextual Recall",
    #     # "Tool Correctness",
    #     # "Contextual Relevancy",
    #     # "Hallucination"
    # ],
    metric_collection="Test",
)
async def meta_agent(query: str):
    update_current_span(
        metadata={"user_id": "11111", "date": "1/1/11"},
        test_case=LLMTestCase(
            input="What is this again?",
            actual_output="this is a latte",
            expected_output="this is a mocha",
            retrieval_context=["I love coffee"],
            context=["I love coffee"],
            tools_called=[ToolCall(name="test")],
            expected_tools=[ToolCall(name="test")],
        ),
    )
    update_current_trace(
        metadata={"input": "input"},
        thread_id="context_thread_id2",
        input="input",
        output="output",
        user_id="111",
        test_case=LLMTestCase(
            input="What is this again?",
            actual_output="this is a latte",
            expected_output="this is a mocha",
            retrieval_context=["I love coffee"],
            context=["I love coffee"],
            tools_called=[ToolCall(name="test")],
            expected_tools=[ToolCall(name="test")],
        ),
    )
    print("query")
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
