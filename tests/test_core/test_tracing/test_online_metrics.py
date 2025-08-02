from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import (
    observe,
    update_current_span,
    update_current_trace,
    TurnContext,
)


@observe(
    type="agent",
    agent_handoffs=["weather_agent", "research_agent", "custom_research_agent"],
    metric_collection="Test",
)
def meta_agent(query: str):
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
        turn_context=TurnContext(
            retrieval_context=[
                "context_retrieval_ dsh oasdfhi uafduasiufhai hd iufa haisu hiucontext",
                "asidufhdsiaufhsiaufhdisaf husai fdisuh isa hfdiuh aiu",
            ],
            tools_called=[ToolCall(name="test")],
        ),
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
    return query


def test_online_metrics():
    for query in [
        "How tall is Mount Everest?",
        "What's the capital of Brazil?",
    ]:
        meta_agent(query)
    assert True
