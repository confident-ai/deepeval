#!/usr/bin/env python3
"""
Test script for imports and basic functionality without requiring API keys.
"""

import sys
import os

# Add the deepeval directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepeval'))

def test_imports():
    """Test all imports work correctly"""
    print("🧪 Testing Enhanced LangGraph Integration Imports...")
    print("=" * 60)
    
    try:
        # Test basic imports
        from deepeval.integrations.langgraph import LangGraphCallbackHandler
        print("LangGraphCallbackHandler import successful")
        
        from deepeval.integrations.langgraph import create_traced_react_agent
        print("create_traced_react_agent import successful")
        
        from deepeval.integrations.langgraph import create_langgraph_callback_config
        print("create_langgraph_callback_config import successful")
        
        from deepeval.integrations.langgraph import extract_graph_metadata
        print("extract_graph_metadata import successful")
        
        from deepeval.integrations.langgraph import trace_langgraph_agent
        print("trace_langgraph_agent import successful")
        
        from deepeval.integrations.langgraph import invoke_traced_agent
        print("invoke_traced_agent import successful")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_callback_handler_creation():
    """Test callback handler creation"""
    print("\n🧪 Testing Callback Handler Creation...")
    
    try:
        from deepeval.integrations.langgraph import LangGraphCallbackHandler
        
        # Test different configurations
        handler1 = LangGraphCallbackHandler()
        print("Default callback handler created")
        
        handler2 = LangGraphCallbackHandler(
            enable_graph_tracing=True,
            enable_node_tracing=False,
            enable_state_tracing=True,
        )
        print("Custom callback handler created")
        
        # Test attributes
        assert hasattr(handler1, 'active_trace_id')
        assert hasattr(handler1, 'metrics')
        assert hasattr(handler1, 'graph_execution_id')
        print("Callback handler has required attributes")
        
        return True
        
    except Exception as e:
        print(f"Callback handler test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\n Testing Utility Functions...")
    
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
        print("Callback config creation successful")
        
        return True
        
    except Exception as e:
        print(f"Utility functions test failed: {e}")
        return False

def test_span_types():
    """Test new span types are available"""
    print("\nTesting New Span Types...")
    
    try:
        from deepeval.tracing.tracing import SpanType
        
        # Check new span types exist
        assert SpanType.GRAPH_EXECUTION.value == "graph_execution"
        assert SpanType.NODE_EXECUTION.value == "node_execution"
        assert SpanType.STATE_TRANSITION.value == "state_transition"
        assert SpanType.CONDITIONAL_ROUTING.value == "conditional_routing"
        assert SpanType.PARALLEL_EXECUTION.value == "parallel_execution"
        
        print("New span types are available")
        
        return True
        
    except Exception as e:
        print(f"Span types test failed: {e}")
        return False

def test_span_classes():
    """Test new span classes are available"""
    print("\nTesting New Span Classes...")
    
    try:
        from deepeval.tracing.types import GraphSpan, NodeSpan, StateTransitionSpan
        
        # Test span class imports
        print("GraphSpan import successful")
        print("NodeSpan import successful")
        print("StateTransitionSpan import successful")
        
        return True
        
    except Exception as e:
        print(f"Span classes test failed: {e}")
        return False

def test_attributes():
    """Test new attributes are available"""
    print("\nTesting New Attributes...")
    
    try:
        from deepeval.tracing.attributes import (
            GraphAttributes, 
            NodeAttributes, 
            StateTransitionAttributes
        )
        
        # Test attribute imports
        print("GraphAttributes import successful")
        print("NodeAttributes import successful")
        print("StateTransitionAttributes import successful")
        
        return True
        
    except Exception as e:
        print(f"Attributes test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced LangGraph Integration Import Test Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        test_imports,
        test_callback_handler_creation,
        test_utility_functions,
        test_span_types,
        test_span_classes,
        test_attributes,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll import tests completed successfully!")
        print("\nWhat's been verified:")
        print("   • All new modules can be imported")
        print("   • Callback handler can be created")
        print("   • Utility functions work")
        print("   • New span types are available")
        print("   • New span classes are available")
        print("   • New attributes are available")
        print("\nNext steps:")
        print("   • Set OPENAI_API_KEY to test with real API calls")
        print("   • Run test_simple.py for full integration testing")
        print("   • Check the TESTING_GUIDE.md for detailed instructions")
    else:
        print(f"\n {total - passed} test(s) failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 