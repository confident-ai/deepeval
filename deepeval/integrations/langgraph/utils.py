"""
Utility functions for LangGraph integration with DeepEval.
"""

from typing import Any, Dict, List, Optional, Union
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent


def extract_graph_metadata(graph: StateGraph) -> Dict[str, Any]:
    """
    Extract metadata from a LangGraph StateGraph for tracing purposes.
    
    Args:
        graph: The LangGraph StateGraph instance
        
    Returns:
        Dictionary containing graph metadata
    """
    metadata = {
        'graph_type': 'StateGraph',
        'node_count': len(graph.nodes),
        'nodes': list(graph.nodes.keys()),
        'edges': [],
        'state_schema': None,
    }
    
    # Extract edges from the graph's edges attribute
    if hasattr(graph, 'edges') and graph.edges:
        for edge in graph.edges:
            if hasattr(edge, 'from_node') and hasattr(edge, 'to_node'):
                metadata['edges'].append({
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'condition': getattr(edge, 'condition', 'default')
                })
    
    # Extract state schema if available
    if hasattr(graph, 'state_schema'):
        metadata['state_schema'] = graph.state_schema
    
    return metadata


def create_langgraph_callback_config(
    metrics: Optional[List[Any]] = None,
    metric_collection: Optional[str] = None,
    enable_graph_tracing: bool = True,
    enable_node_tracing: bool = True,
    enable_state_tracing: bool = True,
) -> Dict[str, Any]:
    """
    Create a callback configuration for LangGraph with DeepEval tracing.
    
    Args:
        metrics: List of DeepEval metrics to apply
        metric_collection: Name for the metric collection
        enable_graph_tracing: Whether to enable graph-level tracing
        enable_node_tracing: Whether to enable node-level tracing
        enable_state_tracing: Whether to enable state transition tracing
        
    Returns:
        Callback configuration dictionary
    """
    from .callback import LangGraphCallbackHandler
    
    callback_handler = LangGraphCallbackHandler(
        metrics=metrics or [],
        metric_collection=metric_collection,
        enable_graph_tracing=enable_graph_tracing,
        enable_node_tracing=enable_node_tracing,
        enable_state_tracing=enable_state_tracing,
    )
    
    return {
        "callbacks": [callback_handler]
    }


def trace_langgraph_agent(
    agent,
    input_data: Union[str, Dict[str, Any]],
    metrics: Optional[List[Any]] = None,
    metric_collection: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Convenience function to trace a LangGraph agent with DeepEval.
    
    Args:
        agent: The LangGraph agent to trace
        input_data: Input data for the agent
        metrics: List of DeepEval metrics to apply
        metric_collection: Name for the metric collection
        **kwargs: Additional arguments to pass to agent.invoke
        
    Returns:
        Agent response
    """
    callback_config = create_langgraph_callback_config(
        metrics=metrics,
        metric_collection=metric_collection
    )
    
    # Prepare input format
    if isinstance(input_data, str):
        input_data = {"messages": [{"role": "user", "content": input_data}]}
    
    # Merge callback config with other kwargs
    config = kwargs.get("config", {})
    config.update(callback_config)
    kwargs["config"] = config
    
    return agent.invoke(input_data, **kwargs)


def create_traced_react_agent(
    model: str,
    tools: List[Any],
    prompt: str,
    metrics: Optional[List[Any]] = None,
    metric_collection: Optional[str] = None,
    **kwargs
):
    """
    Create a React agent with DeepEval tracing enabled.
    
    Args:
        model: The model to use for the agent
        tools: List of tools available to the agent
        prompt: The system prompt for the agent
        metrics: List of DeepEval metrics to apply
        metric_collection: Name for the metric collection
        **kwargs: Additional arguments to pass to create_react_agent
        
    Returns:
        Traced React agent
    """
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        **kwargs
    )
    
    # Store tracing configuration for later use
    agent._deepeval_config = {
        'metrics': metrics,
        'metric_collection': metric_collection
    }
    
    return agent


def invoke_traced_agent(
    agent,
    input_data: Union[str, Dict[str, Any]],
    **kwargs
) -> Any:
    """
    Invoke a traced agent with DeepEval metrics.
    
    Args:
        agent: The traced agent
        input_data: Input data for the agent
        **kwargs: Additional arguments to pass to agent.invoke
        
    Returns:
        Agent response
    """
    if hasattr(agent, '_deepeval_config'):
        config = kwargs.get("config", {})
        callback_config = create_langgraph_callback_config(
            metrics=agent._deepeval_config.get('metrics'),
            metric_collection=agent._deepeval_config.get('metric_collection')
        )
        config.update(callback_config)
        kwargs["config"] = config
    
    return trace_langgraph_agent(agent, input_data, **kwargs) 