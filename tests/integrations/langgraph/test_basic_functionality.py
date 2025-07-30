#!/usr/bin/env python3
"""
Basic functionality test for the enhanced LangGraph integration.
This test doesn't require API keys and focuses on core functionality.
"""

import sys
import os

# Add the deepeval directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepeval'))

def test_basic_imports():
    """Test basic imports work"""
    print("🧪 Testing Basic Imports...")
    
    try:
        from deepeval.integrations.langgraph import LangGraphCallbackHandler
        print("✅ LangGraphCallbackHandler import successful")
        
        from deepeval.integrations.langgraph import create_langgraph_callback_config
        print("✅ create_langgraph_callback_config import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_callback_handler():
    """Test callback handler creation and basic functionality"""
    print("\n🧪 Testing Callback Handler...")
    
    try:
        from deepeval.integrations.langgraph import LangGraphCallbackHandler
        
        # Create handler
        handler = LangGraphCallbackHandler(
            enable_graph_tracing=True,
            enable_node_tracing=True,
            enable_state_tracing=True,
        )
        
        print("✅ Callback handler created successfully")
        
        # Test basic attributes
        assert hasattr(handler, 'active_trace_id')
        assert hasattr(handler, 'metrics')
        assert hasattr(handler, 'graph_execution_id')
        assert hasattr(handler, 'current_node')
        assert hasattr(handler, 'node_execution_order')
        
        print("✅ Callback handler has all required attributes")
        
        return True
    except Exception as e:
        print(f"❌ Callback handler test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\n🧪 Testing Utility Functions...")
    
    try:
        from deepeval.integrations.langgraph import create_langgraph_callback_config, LangGraphCallbackHandler
        
        # Test config creation
        config = create_langgraph_callback_config(
            metric_collection="test_collection",
            enable_graph_tracing=True,
            enable_node_tracing=False,
            enable_state_tracing=True,
        )
        
        assert "callbacks" in config
        assert len(config["callbacks"]) == 1
        assert isinstance(config["callbacks"][0], LangGraphCallbackHandler)
        
        print("✅ Utility functions work correctly")
        
        return True
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def test_span_types():
    """Test new span types are available"""
    print("\n🧪 Testing New Span Types...")
    
    try:
        from deepeval.tracing.tracing import SpanType
        
        # Check new span types exist
        expected_types = [
            "graph_execution",
            "node_execution", 
            "state_transition",
            "conditional_routing",
            "parallel_execution"
        ]
        
        for span_type in expected_types:
            assert hasattr(SpanType, span_type.upper().replace('_', '_'))
            print(f"✅ {span_type} span type available")
        
        return True
    except Exception as e:
        print(f"❌ Span types test failed: {e}")
        return False

def test_span_classes():
    """Test new span classes are available"""
    print("\n🧪 Testing New Span Classes...")
    
    try:
        from deepeval.tracing.types import GraphSpan, NodeSpan, StateTransitionSpan
        
        # Test class imports
        print("✅ GraphSpan class available")
        print("✅ NodeSpan class available") 
        print("✅ StateTransitionSpan class available")
        
        return True
    except Exception as e:
        print(f"❌ Span classes test failed: {e}")
        return False

def test_attributes():
    """Test new attributes are available"""
    print("\n🧪 Testing New Attributes...")
    
    try:
        from deepeval.tracing.attributes import (
            GraphAttributes, 
            NodeAttributes, 
            StateTransitionAttributes
        )
        
        # Test attribute imports
        print("✅ GraphAttributes available")
        print("✅ NodeAttributes available")
        print("✅ StateTransitionAttributes available")
        
        return True
    except Exception as e:
        print(f"❌ Attributes test failed: {e}")
        return False

def test_langgraph_compatibility():
    """Test LangGraph compatibility"""
    print("\n🧪 Testing LangGraph Compatibility...")
    
    try:
        import langgraph
        print(f"✅ LangGraph imported successfully")
        
        from langgraph.graph import StateGraph
        from typing import Dict, Any
        
        # Test basic StateGraph creation (without execution)
        workflow = StateGraph(state_schema=Dict[str, Any])
        print("✅ StateGraph creation successful")
        
        return True
    except Exception as e:
        print(f"❌ LangGraph compatibility test failed: {e}")
        return False

def main():
    """Run all basic functionality tests"""
    print("🚀 Enhanced LangGraph Integration - Basic Functionality Test")
    print("=" * 70)
    
    # Run tests
    tests = [
        test_basic_imports,
        test_callback_handler,
        test_utility_functions,
        test_span_types,
        test_span_classes,
        test_attributes,
        test_langgraph_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic functionality tests completed successfully!")
        print("\n📋 What's been verified:")
        print("   • All new modules can be imported")
        print("   • Callback handler can be created and configured")
        print("   • Utility functions work correctly")
        print("   • New span types are available")
        print("   • New span classes are available")
        print("   • New attributes are available")
        print("   • LangGraph compatibility is maintained")
        print("\n💡 Next steps:")
        print("   • Set OPENAI_API_KEY to test with real API calls")
        print("   • Run test_simple.py for full integration testing")
        print("   • Check the TESTING_GUIDE.md for detailed instructions")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 