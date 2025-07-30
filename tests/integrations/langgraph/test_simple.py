#!/usr/bin/env python3
"""
Simple test script for the enhanced LangGraph integration.
Run this to quickly test the basic functionality.
"""

import os
import sys
import time

# Add the deepeval directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepeval'))

def test_basic_functionality():
    """Test basic LangGraph integration functionality"""
    print("🧪 Testing Enhanced LangGraph Integration...")
    print("=" * 50)
    
    try:
        # Import required modules
        from deepeval.integrations.langgraph import LangGraphCallbackHandler
        from langgraph.prebuilt import create_react_agent
        
        print("✅ Imports successful")
        
        # Define a simple tool
        def get_weather(city: str) -> str:
            return f"It's always sunny in {city}!"
        
        print("✅ Tool function defined")
        
        # Create agent
        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            prompt="You are a helpful assistant that provides weather information.",
        )
        
        print("✅ Agent created successfully")
        
        # Create enhanced callback handler
        callback_handler = LangGraphCallbackHandler(
            enable_graph_tracing=True,
            enable_node_tracing=True,
            enable_state_tracing=True,
        )
        
        print("✅ Enhanced callback handler created")
        
        # Test invocation
        print("🔄 Testing agent invocation...")
        result = agent.invoke(
            input={"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
            config={"callbacks": [callback_handler]}
        )
        
        print(f"✅ Agent response: {result['messages'][-1]['content'][:100]}...")
        print("✅ Basic functionality test passed!")
        
        # Wait for traces to be processed
        print("⏳ Waiting for traces to be processed...")
        time.sleep(3)
        print("✅ Traces should be available on Confident AI")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have installed the dependencies:")
        print("   pip install langgraph openai")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_traced_agent():
    """Test traced agent creation"""
    print("\n🧪 Testing Traced Agent Creation...")
    
    try:
        from deepeval.integrations.langgraph import create_traced_react_agent
        from deepeval.metrics import TaskCompletionMetric
        
        def get_weather(city: str) -> str:
            return f"It's always sunny in {city}!"
        
        # Create traced agent
        agent = create_traced_react_agent(
            model="openai:gpt-4o-mini",
            tools=[get_weather],
            prompt="You are a helpful assistant.",
            metrics=[TaskCompletionMetric(threshold=0.7)],
            metric_collection="test_collection"
        )
        
        # Verify agent has tracing config
        assert hasattr(agent, '_deepeval_config')
        assert agent._deepeval_config['metric_collection'] == "test_collection"
        
        print("✅ Traced agent creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Traced agent test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\n🧪 Testing Utility Functions...")
    
    try:
        from deepeval.integrations.langgraph import create_langgraph_callback_config
        
        # Test callback config creation
        config = create_langgraph_callback_config(
            metric_collection="test_utils",
            enable_graph_tracing=True,
            enable_node_tracing=False,
            enable_state_tracing=True,
        )
        
        assert "callbacks" in config
        assert len(config["callbacks"]) == 1
        
        print("✅ Utility functions test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced LangGraph Integration Test Suite")
    print("=" * 60)
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it and try again.")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print("✅ OpenAI API key found")
    
    # Run tests
    tests = [
        test_basic_functionality,
        test_traced_agent,
        test_utility_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests completed successfully!")
        print("\n📊 Check your Confident AI dashboard to see:")
        print("   • Graph execution traces")
        print("   • Node-level execution details")
        print("   • State transition tracking")
        print("   • Metric evaluation results")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 