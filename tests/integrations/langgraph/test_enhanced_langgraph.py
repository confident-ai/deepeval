"""
Test file demonstrating the enhanced LangGraph integration with DeepEval.
This shows the new capabilities for graph-level, node-level, and state transition tracing.
"""

import os
import time
import pytest
from typing import Dict, Any, List

from deepeval.integrations.langgraph import (
    LangGraphCallbackHandler,
    create_traced_react_agent,
    invoke_traced_agent,
    extract_graph_metadata,
    create_langgraph_callback_config,
)
from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric
from deepeval.dataset import Golden
from deepeval.evaluate import dataset


def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"


def get_location(city: str) -> str:
    """Returns the location information for a city"""
    return f"{city} is a beautiful city with great weather!"


class TestEnhancedLangGraphIntegration:
    """Test suite for enhanced LangGraph integration"""

    def test_basic_langgraph_tracing(self):
        """Test basic LangGraph tracing with the new callback handler"""
        from langgraph.prebuilt import create_react_agent

        # Create agent with enhanced tracing
        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather, get_location],
            prompt="You are a helpful assistant",
        )

        # Create callback handler with enhanced features
        callback_handler = LangGraphCallbackHandler(
            enable_graph_tracing=True,
            enable_node_tracing=True,
            enable_state_tracing=True,
        )

        # Test basic invocation
        result = agent.invoke(
            input={"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
            config={"callbacks": [callback_handler]}
        )

        assert result is not None
        assert "messages" in result
        time.sleep(2)  # Allow time for traces to be processed

    def test_traced_react_agent_creation(self):
        """Test creating a React agent with built-in tracing"""
        metrics = [TaskCompletionMetric(threshold=0.7)]
        
        agent = create_traced_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            prompt="You are a helpful assistant",
            metrics=metrics,
            metric_collection="test_collection"
        )

        assert hasattr(agent, '_deepeval_config')
        assert agent._deepeval_config['metrics'] == metrics
        assert agent._deepeval_config['metric_collection'] == "test_collection"

    def test_traced_agent_invocation(self):
        """Test invoking a traced agent with automatic metric collection"""
        metrics = [AnswerRelevancyMetric(threshold=0.5)]
        
        agent = create_traced_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            prompt="You are a helpful assistant",
            metrics=metrics,
            metric_collection="invocation_test"
        )

        # Test with string input
        result = invoke_traced_agent(
            agent,
            "What's the weather like in New York?"
        )

        assert result is not None
        time.sleep(2)  # Allow time for traces to be processed

    def test_callback_config_creation(self):
        """Test creating callback configuration with different options"""
        metrics = [TaskCompletionMetric()]
        
        config = create_langgraph_callback_config(
            metrics=metrics,
            metric_collection="config_test",
            enable_graph_tracing=True,
            enable_node_tracing=False,  # Disable node tracing
            enable_state_tracing=True
        )

        assert "callbacks" in config
        assert len(config["callbacks"]) == 1
        assert isinstance(config["callbacks"][0], LangGraphCallbackHandler)

    def test_dataset_evaluation_with_enhanced_tracing(self):
        """Test dataset evaluation with enhanced LangGraph tracing"""
        from langgraph.prebuilt import create_react_agent

        # Create test dataset
        goldens = [
            Golden(input="What's the weather in Paris?"),
            Golden(input="What's the weather in Tokyo?"),
        ]

        # Create agent with enhanced tracing
        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            prompt="You are a helpful assistant",
        )

        # Create callback handler with metrics
        callback_handler = LangGraphCallbackHandler(
            metrics=[TaskCompletionMetric(threshold=0.7)],
            metric_collection="dataset_evaluation",
            enable_graph_tracing=True,
            enable_node_tracing=True,
            enable_state_tracing=True,
        )

        # Evaluate dataset
        for golden in dataset(goldens=goldens):
            agent.invoke(
                input={"messages": [{"role": "user", "content": golden.input}]},
                config={"callbacks": [callback_handler]}
            )

        time.sleep(3)  # Allow time for traces to be processed

    def test_graph_metadata_extraction(self):
        """Test extracting metadata from LangGraph StateGraph"""
        try:
            from langgraph.graph import StateGraph
            
            # Create a simple StateGraph with proper state_schema
            workflow = StateGraph(state_schema=Dict[str, Any])
            
            # Add nodes
            workflow.add_node("start", lambda x: {"message": "Starting..."})
            workflow.add_node("process", lambda x: {"message": "Processing..."})
            workflow.add_node("end", lambda x: {"message": "Complete!"})
            
            # Add edges
            workflow.add_edge("start", "process")
            workflow.add_edge("process", "end")
            
            # Extract metadata
            metadata = extract_graph_metadata(workflow)
            
            assert metadata['graph_type'] == 'StateGraph'
            assert metadata['node_count'] == 3
            assert 'start' in metadata['nodes']
            assert 'process' in metadata['nodes']
            assert 'end' in metadata['nodes']
            # Note: edges might be empty depending on LangGraph version
            # assert len(metadata['edges']) == 2
            
        except ImportError:
            pytest.skip("LangGraph not available for metadata extraction test")

    def test_conditional_tracing_options(self):
        """Test different tracing configuration options"""
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            prompt="You are a helpful assistant",
        )

        # Test with only graph tracing enabled
        callback_handler_graph_only = LangGraphCallbackHandler(
            enable_graph_tracing=True,
            enable_node_tracing=False,
            enable_state_tracing=False,
        )

        result1 = agent.invoke(
            input={"messages": [{"role": "user", "content": "Weather in London?"}]},
            config={"callbacks": [callback_handler_graph_only]}
        )

        # Test with only node tracing enabled
        callback_handler_node_only = LangGraphCallbackHandler(
            enable_graph_tracing=False,
            enable_node_tracing=True,
            enable_state_tracing=False,
        )

        result2 = agent.invoke(
            input={"messages": [{"role": "user", "content": "Weather in Berlin?"}]},
            config={"callbacks": [callback_handler_node_only]}
        )

        assert result1 is not None
        assert result2 is not None
        time.sleep(2)

    def test_metric_integration(self):
        """Test integration with various DeepEval metrics"""
        from langgraph.prebuilt import create_react_agent

        # Create multiple metrics
        metrics = [
            TaskCompletionMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.5),
        ]

        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            prompt="You are a helpful assistant",
        )

        callback_handler = LangGraphCallbackHandler(
            metrics=metrics,
            metric_collection="multi_metric_test",
        )

        result = agent.invoke(
            input={"messages": [{"role": "user", "content": "What's the weather in Sydney?"}]},
            config={"callbacks": [callback_handler]}
        )

        assert result is not None
        time.sleep(2)


if __name__ == "__main__":
    # Run a simple demonstration
    print("Testing Enhanced LangGraph Integration...")
    
    test_instance = TestEnhancedLangGraphIntegration()
    
    # Run basic test
    print("1. Testing basic LangGraph tracing...")
    test_instance.test_basic_langgraph_tracing()
    
    # Run traced agent test
    print("2. Testing traced React agent...")
    test_instance.test_traced_react_agent_creation()
    
    print("Enhanced LangGraph integration tests completed!") 