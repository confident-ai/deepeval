"""
Enhanced LangGraph Integration Example

This example demonstrates the new capabilities of DeepEval's enhanced LangGraph integration,
including graph-level, node-level, and state transition tracing.
"""

import os
import time
from typing import Dict, Any, List

from deepeval.integrations.langgraph import (
    LangGraphCallbackHandler,
    create_traced_react_agent,
    invoke_traced_agent,
    extract_graph_metadata,
    create_langgraph_callback_config,
)
from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.dataset import Golden
from deepeval.evaluate import dataset


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


def get_location(city: str) -> str:
    """Returns the location information for a city"""
    return f"{city} is a beautiful city with great weather!"


def get_population(city: str) -> str:
    """Returns the population information for a city"""
    return f"{city} has a population of approximately 1 million people."


def demonstrate_basic_tracing():
    """Demonstrate basic LangGraph tracing capabilities"""
    print("=== Basic LangGraph Tracing ===")
    
    from langgraph.prebuilt import create_react_agent

    # Create agent with multiple tools
    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, get_location, get_population],
        prompt="You are a helpful assistant that provides information about cities.",
    )

    # Create enhanced callback handler
    callback_handler = LangGraphCallbackHandler(
        enable_graph_tracing=True,
        enable_node_tracing=True,
        enable_state_tracing=True,
    )

    # Test basic invocation
    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "Tell me about San Francisco"}]},
        config={"callbacks": [callback_handler]}
    )

    print(f"Agent response: {result['messages'][-1]['content'][:100]}...")
    print("✓ Basic tracing completed - check Confident AI for traces\n")


def demonstrate_traced_agent_creation():
    """Demonstrate creating agents with built-in tracing"""
    print("=== Traced Agent Creation ===")
    
    # Create metrics for evaluation
    metrics = [
        TaskCompletionMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.5),
    ]
    
    # Create agent with built-in tracing
    agent = create_traced_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, get_location],
        prompt="You are a helpful assistant that provides city information.",
        metrics=metrics,
        metric_collection="city_info_agent"
    )

    print(f"Agent created with {len(metrics)} metrics")
    print(f"Metric collection: {agent._deepeval_config['metric_collection']}")
    print("✓ Traced agent creation completed\n")


def demonstrate_traced_agent_invocation():
    """Demonstrate invoking traced agents with automatic metric collection"""
    print("=== Traced Agent Invocation ===")
    
    # Create agent with tracing
    agent = create_traced_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, get_location],
        prompt="You are a helpful assistant.",
        metrics=[TaskCompletionMetric(threshold=0.7)],
        metric_collection="invocation_demo"
    )

    # Test with string input (automatically converted to proper format)
    result = invoke_traced_agent(
        agent,
        "What's the weather like in New York and tell me about the city?"
    )

    print(f"Agent response: {result['messages'][-1]['content'][:100]}...")
    print("✓ Traced agent invocation completed - metrics automatically collected\n")


def demonstrate_callback_configuration():
    """Demonstrate different callback configuration options"""
    print("=== Callback Configuration Options ===")
    
    # Create different configurations
    configs = [
        {
            "name": "Full Tracing",
            "config": create_langgraph_callback_config(
                enable_graph_tracing=True,
                enable_node_tracing=True,
                enable_state_tracing=True,
            )
        },
        {
            "name": "Graph Only",
            "config": create_langgraph_callback_config(
                enable_graph_tracing=True,
                enable_node_tracing=False,
                enable_state_tracing=False,
            )
        },
        {
            "name": "Node Only",
            "config": create_langgraph_callback_config(
                enable_graph_tracing=False,
                enable_node_tracing=True,
                enable_state_tracing=False,
            )
        },
    ]

    from langgraph.prebuilt import create_react_agent
    
    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather],
        prompt="You are a helpful assistant.",
    )

    for config_info in configs:
        print(f"Testing {config_info['name']}...")
        result = agent.invoke(
            input={"messages": [{"role": "user", "content": "Weather in Paris?"}]},
            config=config_info['config']
        )
        print(f"  ✓ {config_info['name']} completed")

    print("✓ All callback configurations tested\n")


def demonstrate_dataset_evaluation():
    """Demonstrate dataset evaluation with enhanced tracing"""
    print("=== Dataset Evaluation with Enhanced Tracing ===")
    
    # Create test dataset
    goldens = [
        Golden(input="What's the weather in Tokyo?"),
        Golden(input="Tell me about London"),
        Golden(input="What's the population of Berlin?"),
    ]

    from langgraph.prebuilt import create_react_agent

    # Create agent with enhanced tracing
    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, get_location, get_population],
        prompt="You are a helpful assistant that provides city information.",
    )

    # Create callback handler with metrics
    callback_handler = LangGraphCallbackHandler(
        metrics=[
            TaskCompletionMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.5),
        ],
        metric_collection="dataset_evaluation_demo",
        enable_graph_tracing=True,
        enable_node_tracing=True,
        enable_state_tracing=True,
    )

    # Evaluate dataset
    print(f"Evaluating {len(goldens)} test cases...")
    for i, golden in enumerate(dataset(goldens=goldens), 1):
        print(f"  Processing test case {i}/{len(goldens)}: {golden.input}")
        agent.invoke(
            input={"messages": [{"role": "user", "content": golden.input}]},
            config={"callbacks": [callback_handler]}
        )

    print("✓ Dataset evaluation completed - check Confident AI for results\n")


def demonstrate_graph_metadata_extraction():
    """Demonstrate extracting metadata from LangGraph StateGraph"""
    print("=== Graph Metadata Extraction ===")
    
    try:
        from langgraph.graph import StateGraph
        
        # Create a simple StateGraph
        workflow = StateGraph(StateType=Dict[str, Any])
        
        # Add nodes
        workflow.add_node("start", lambda x: {"message": "Starting..."})
        workflow.add_node("process", lambda x: {"message": "Processing..."})
        workflow.add_node("end", lambda x: {"message": "Complete!"})
        
        # Add edges
        workflow.add_edge("start", "process")
        workflow.add_edge("process", "end")
        
        # Extract metadata
        metadata = extract_graph_metadata(workflow)
        
        print(f"Graph type: {metadata['graph_type']}")
        print(f"Node count: {metadata['node_count']}")
        print(f"Nodes: {metadata['nodes']}")
        print(f"Edges: {metadata['edges']}")
        print("✓ Graph metadata extraction completed\n")
        
    except ImportError:
        print("LangGraph not available for metadata extraction demo\n")


def demonstrate_advanced_metrics():
    """Demonstrate integration with advanced DeepEval metrics"""
    print("=== Advanced Metrics Integration ===")
    
    from langgraph.prebuilt import create_react_agent

    # Create multiple metrics for comprehensive evaluation
    metrics = [
        TaskCompletionMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.5),
        ContextualRelevancyMetric(threshold=0.6),
    ]

    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather, get_location],
        prompt="You are a helpful assistant that provides detailed city information.",
    )

    callback_handler = LangGraphCallbackHandler(
        metrics=metrics,
        metric_collection="advanced_metrics_demo",
    )

    # Test with complex query
    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "What's the weather like in Sydney and what makes it special?"}]},
        config={"callbacks": [callback_handler]}
    )

    print(f"Agent response: {result['messages'][-1]['content'][:100]}...")
    print(f"Applied {len(metrics)} different metrics for evaluation")
    print("✓ Advanced metrics integration completed\n")


def main():
    """Run all demonstrations"""
    print("🚀 Enhanced LangGraph Integration Demo")
    print("=" * 50)
    
    # Set up environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Run all demonstrations
        demonstrate_basic_tracing()
        demonstrate_traced_agent_creation()
        demonstrate_traced_agent_invocation()
        demonstrate_callback_configuration()
        demonstrate_dataset_evaluation()
        demonstrate_graph_metadata_extraction()
        demonstrate_advanced_metrics()
        
        print("🎉 All demonstrations completed successfully!")
        print("\n📊 Check your Confident AI dashboard to see:")
        print("   • Graph execution traces")
        print("   • Node-level execution details")
        print("   • State transition tracking")
        print("   • Metric evaluation results")
        print("   • Performance insights")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main() 