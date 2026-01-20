"""
Demo script to run all LangGraph apps sequentially with DeepEval tracing.
Run with: python -m tests.test_integrations.test_langgraph.apps.main
"""

import asyncio
from langchain_core.messages import HumanMessage
from deepeval.integrations.langchain import CallbackHandler


def separator(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def main():
    # 1. Simple App
    separator("1. SIMPLE APP - Single Tool (Weather)")
    from tests.test_integrations.test_langgraph.apps.langgraph_simple_app import app as simple_app
    
    callback = CallbackHandler(
        name="demo-simple",
        tags=["demo", "simple"],
        metadata={"app": "simple"},
    )
    result = simple_app.invoke(
        {"messages": [HumanMessage(content="What's the weather in San Francisco?")]},
        config={"callbacks": [callback]},
    )
    print(f"Response: {result['messages'][-1].content}")

    # 2. Multiple Tools App
    separator("2. MULTIPLE TOOLS APP - Weather, Population, Timezone, Calculator")
    from tests.test_integrations.test_langgraph.apps.langgraph_multiple_tools_app import app as multiple_tools_app
    
    callback = CallbackHandler(
        name="demo-multiple-tools",
        tags=["demo", "multiple-tools"],
        metadata={"app": "multiple_tools"},
    )
    result = multiple_tools_app.invoke(
        {"messages": [HumanMessage(content="Tell me about Tokyo - weather, population, and timezone. Also calculate 15 * 23.")]},
        config={"callbacks": [callback]},
    )
    print(f"Response: {result['messages'][-1].content}")

    # 3. Streaming App (Sync)
    separator("3. STREAMING APP - Stock Price Tools")
    from tests.test_integrations.test_langgraph.apps.langgraph_streaming_app import sync_app as streaming_app
    
    callback = CallbackHandler(
        name="demo-streaming",
        tags=["demo", "streaming"],
        metadata={"app": "streaming"},
    )
    print("Streaming chunks:")
    for chunk in streaming_app.stream(
        {"messages": [HumanMessage(content="What's the stock price of AAPL?")]},
        config={"callbacks": [callback]},
    ):
        print(f"  Chunk: {list(chunk.keys())}")
    
    # Also get final result
    callback = CallbackHandler(
        name="demo-streaming-invoke",
        tags=["demo", "streaming"],
        metadata={"app": "streaming"},
    )
    result = streaming_app.invoke(
        {"messages": [HumanMessage(content="What's the stock price of MSFT?")]},
        config={"callbacks": [callback]},
    )
    print(f"Final Response: {result['messages'][-1].content}")

    # 4. Conditional App
    separator("4. CONDITIONAL APP - Intent-Based Routing")
    from tests.test_integrations.test_langgraph.apps.langgraph_conditional_app import app as conditional_app
    
    # Research route
    callback = CallbackHandler(
        name="demo-conditional-research",
        tags=["demo", "conditional", "research"],
        metadata={"app": "conditional", "intent": "research"},
    )
    result = conditional_app.invoke(
        {"messages": [HumanMessage(content="Research information about AI")]},
        config={"callbacks": [callback]},
    )
    print(f"Research Response: {result['messages'][-1].content}")
    
    # Fact check route
    callback = CallbackHandler(
        name="demo-conditional-factcheck",
        tags=["demo", "conditional", "factcheck"],
        metadata={"app": "conditional", "intent": "factcheck"},
    )
    result = conditional_app.invoke(
        {"messages": [HumanMessage(content="Fact check: The earth is round")]},
        config={"callbacks": [callback]},
    )
    print(f"Fact Check Response: {result['messages'][-1].content}")

    # 5. Parallel Tools App
    separator("5. PARALLEL TOOLS APP - Multiple Parallel Tool Calls")
    from tests.test_integrations.test_langgraph.apps.langgraph_parallel_tools_app import sync_app as parallel_app
    
    callback = CallbackHandler(
        name="demo-parallel",
        tags=["demo", "parallel"],
        metadata={"app": "parallel"},
    )
    result = parallel_app.invoke(
        {"messages": [HumanMessage(content="Get weather for Tokyo, New York, and London.")]},
        config={"callbacks": [callback]},
    )
    print(f"Response: {result['messages'][-1].content}")

    # 6. Async App (run synchronously for demo)
    separator("6. ASYNC APP - Database Search & Translation")
    from tests.test_integrations.test_langgraph.apps.langgraph_async_app import app as async_app
    
    async def run_async():
        callback = CallbackHandler(
            name="demo-async",
            tags=["demo", "async"],
            metadata={"app": "async"},
        )
        result = await async_app.ainvoke(
            {"messages": [HumanMessage(content="Search for information about Python")]},
            config={"callbacks": [callback]},
        )
        return result
    
    result = asyncio.run(run_async())
    print(f"Response: {result['messages'][-1].content}")

    # 7. Multi-Turn App
    separator("7. MULTI-TURN APP - Shopping Cart with Memory")
    from tests.test_integrations.test_langgraph.apps.langgraph_multi_turn_app import get_app_with_memory
    
    app = get_app_with_memory()
    thread_id = "demo-session-001"
    
    # Turn 1
    callback = CallbackHandler(
        name="demo-multi-turn-1",
        tags=["demo", "multi-turn", "turn-1"],
        metadata={"app": "multi_turn", "turn": 1},
        thread_id=thread_id,
        user_id="demo-user",
    )
    result = app.invoke(
        {"messages": [HumanMessage(content="Add 2 apples to my cart")]},
        config={"callbacks": [callback], "configurable": {"thread_id": thread_id}},
    )
    print(f"Turn 1: {result['messages'][-1].content}")
    
    # Turn 2
    callback = CallbackHandler(
        name="demo-multi-turn-2",
        tags=["demo", "multi-turn", "turn-2"],
        metadata={"app": "multi_turn", "turn": 2},
        thread_id=thread_id,
        user_id="demo-user",
    )
    result = app.invoke(
        {"messages": [HumanMessage(content="Also add 3 oranges")]},
        config={"callbacks": [callback], "configurable": {"thread_id": thread_id}},
    )
    print(f"Turn 2: {result['messages'][-1].content}")
    
    # Turn 3
    callback = CallbackHandler(
        name="demo-multi-turn-3",
        tags=["demo", "multi-turn", "turn-3"],
        metadata={"app": "multi_turn", "turn": 3},
        thread_id=thread_id,
        user_id="demo-user",
    )
    result = app.invoke(
        {"messages": [HumanMessage(content="What's in my cart?")]},
        config={"callbacks": [callback], "configurable": {"thread_id": thread_id}},
    )
    print(f"Turn 3: {result['messages'][-1].content}")

    separator("ALL DEMOS COMPLETE")


if __name__ == "__main__":
    main()

