# Enhanced LangGraph Integration

This module provides enhanced tracing and evaluation capabilities for LangGraph applications, building upon DeepEval's existing tracing infrastructure to provide comprehensive observability for graph-based workflows.

## Features

### 🎯 **Graph-Level Tracing**
- Track complete graph execution from start to finish
- Monitor graph compilation and configuration
- Capture execution mode (sequential, parallel, conditional)

### 🔍 **Node-Level Tracing**
- Individual node execution tracking
- Node dependencies and execution order
- Conditional logic and parallel group monitoring
- Performance metrics per node

### 🔄 **State Transition Tracking**
- Monitor state changes between nodes
- Track routing decisions and conditions
- Capture state snapshots for debugging

### 📊 **Enhanced Observability**
- Visual graph execution traces on Confident AI
- Performance bottlenecks identification
- Cost analysis and optimization insights
- Comprehensive debugging tools

## Quick Start

### Basic Usage

```python
from deepeval.integrations.langgraph import LangGraphCallbackHandler
from langgraph.prebuilt import create_react_agent

# Create agent
agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[your_tools],
    prompt="You are a helpful assistant",
)

# Create enhanced callback handler
callback_handler = LangGraphCallbackHandler(
    enable_graph_tracing=True,
    enable_node_tracing=True,
    enable_state_tracing=True,
)

# Use with agent
result = agent.invoke(
    input={"messages": [{"role": "user", "content": "Your query"}]},
    config={"callbacks": [callback_handler]}
)
```

### Traced Agent Creation

```python
from deepeval.integrations.langgraph import create_traced_react_agent
from deepeval.metrics import TaskCompletionMetric

# Create agent with built-in tracing
agent = create_traced_react_agent(
    model="openai:gpt-4o-mini",
    tools=[your_tools],
    prompt="You are a helpful assistant",
    metrics=[TaskCompletionMetric(threshold=0.7)],
    metric_collection="my_agent"
)

# Invoke with automatic tracing
result = agent.invoke(
    input={"messages": [{"role": "user", "content": "Your query"}]}
)
```

### Convenience Functions

```python
from deepeval.integrations.langgraph import invoke_traced_agent

# Simple invocation with string input
result = invoke_traced_agent(
    agent,
    "What's the weather like in San Francisco?"
)
```

## API Reference

### LangGraphCallbackHandler

Enhanced callback handler specifically designed for LangGraph applications.

#### Parameters

- `metrics` (List[BaseMetric], optional): List of DeepEval metrics to apply
- `metric_collection` (str, optional): Name for the metric collection
- `enable_graph_tracing` (bool): Enable graph-level tracing (default: True)
- `enable_node_tracing` (bool): Enable node-level tracing (default: True)
- `enable_state_tracing` (bool): Enable state transition tracing (default: True)

#### Methods

- `on_graph_start()`: Track graph execution start
- `on_graph_end()`: Track graph execution end
- `on_node_start()`: Track individual node execution start
- `on_node_end()`: Track individual node execution end
- `on_state_transition()`: Track state transitions between nodes

### Utility Functions

#### `create_langgraph_callback_config()`

Create a callback configuration for LangGraph with DeepEval tracing.

```python
config = create_langgraph_callback_config(
    metrics=[TaskCompletionMetric()],
    metric_collection="my_collection",
    enable_graph_tracing=True,
    enable_node_tracing=True,
    enable_state_tracing=True,
)
```

#### `create_traced_react_agent()`

Create a React agent with DeepEval tracing enabled.

```python
agent = create_traced_react_agent(
    model="openai:gpt-4o-mini",
    tools=[your_tools],
    prompt="You are a helpful assistant",
    metrics=[TaskCompletionMetric()],
    metric_collection="my_agent"
)
```

#### `invoke_traced_agent()`

Invoke a traced agent with automatic metric collection.

```python
result = invoke_traced_agent(
    agent,
    "Your query here"
)
```

#### `extract_graph_metadata()`

Extract metadata from a LangGraph StateGraph for tracing purposes.

```python
metadata = extract_graph_metadata(your_state_graph)
```

## Span Types

The enhanced integration introduces new span types for better LangGraph observability:

### GraphSpan
- **Purpose**: Track complete graph execution
- **Attributes**: Graph configuration, node count, execution mode
- **Metadata**: Graph compilation time, execution strategy

### NodeSpan
- **Purpose**: Track individual node execution
- **Attributes**: Node type, dependencies, execution order
- **Metadata**: Conditional logic, parallel group information

### StateTransitionSpan
- **Purpose**: Track state transitions between nodes
- **Attributes**: From/to nodes, state changes, routing decisions
- **Metadata**: Transition conditions, routing logic

## Configuration Options

### Tracing Levels

You can configure different levels of tracing granularity:

```python
# Full tracing (default)
callback_handler = LangGraphCallbackHandler(
    enable_graph_tracing=True,
    enable_node_tracing=True,
    enable_state_tracing=True,
)

# Graph-level only
callback_handler = LangGraphCallbackHandler(
    enable_graph_tracing=True,
    enable_node_tracing=False,
    enable_state_tracing=False,
)

# Node-level only
callback_handler = LangGraphCallbackHandler(
    enable_graph_tracing=False,
    enable_node_tracing=True,
    enable_state_tracing=False,
)
```

### Metric Integration

```python
from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric

# Single metric
callback_handler = LangGraphCallbackHandler(
    metrics=[TaskCompletionMetric(threshold=0.7)]
)

# Multiple metrics
callback_handler = LangGraphCallbackHandler(
    metrics=[
        TaskCompletionMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.5),
    ],
    metric_collection="comprehensive_evaluation"
)
```

## Dataset Evaluation

```python
from deepeval.dataset import Golden
from deepeval.evaluate import dataset

# Create test dataset
goldens = [
    Golden(input="What's the weather in Tokyo?"),
    Golden(input="Tell me about London"),
]

# Create callback handler with metrics
callback_handler = LangGraphCallbackHandler(
    metrics=[TaskCompletionMetric(threshold=0.7)],
    metric_collection="dataset_evaluation",
)

# Evaluate dataset
for golden in dataset(goldens=goldens):
    agent.invoke(
        input={"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [callback_handler]}
    )
```

## Visualization

All traces are automatically sent to Confident AI for visualization:

- **Graph Execution View**: See the complete graph execution flow
- **Node Performance**: Monitor individual node performance
- **State Transitions**: Visualize state changes and routing decisions
- **Metric Results**: View evaluation results for each component
- **Performance Insights**: Identify bottlenecks and optimization opportunities

## Best Practices

### 1. **Choose Appropriate Tracing Levels**
- Use full tracing for development and debugging
- Use selective tracing for production monitoring
- Disable unnecessary tracing to minimize overhead

### 2. **Configure Meaningful Metrics**
- Select metrics relevant to your use case
- Use different metrics for different components
- Set appropriate thresholds for your requirements

### 3. **Monitor Performance**
- Track execution times for different nodes
- Monitor memory usage and token consumption
- Use performance insights for optimization

### 4. **Debugging Workflows**
- Use state transition tracking to understand routing decisions
- Monitor conditional logic execution
- Analyze graph execution patterns

## Examples

See the `examples/enhanced_langgraph_example.py` file for comprehensive examples demonstrating all features.

## Migration from Basic Integration

If you're currently using the basic LangGraph integration, here's how to upgrade:

### Before (Basic Integration)
```python
from deepeval.integrations.langchain.callback import CallbackHandler

callback_handler = CallbackHandler()
result = agent.invoke(input, config={"callbacks": [callback_handler]})
```

### After (Enhanced Integration)
```python
from deepeval.integrations.langgraph import LangGraphCallbackHandler

callback_handler = LangGraphCallbackHandler(
    enable_graph_tracing=True,
    enable_node_tracing=True,
    enable_state_tracing=True,
)
result = agent.invoke(input, config={"callbacks": [callback_handler]})
```

## Troubleshooting

### Common Issues

1. **Traces not appearing in Confident AI**
   - Ensure you're logged into Confident AI
   - Check that tracing is enabled
   - Verify callback handler is properly configured

2. **Performance overhead**
   - Disable unnecessary tracing levels
   - Use sampling for high-traffic applications
   - Monitor execution times

3. **Missing node information**
   - Ensure `enable_node_tracing=True`
   - Check that nodes are properly configured
   - Verify LangGraph version compatibility

### Support

For issues and questions:
- Check the [DeepEval documentation](https://deepeval.com/docs)
- Visit the [Confident AI platform](https://confident-ai.com)
- Join the [DeepEval community](https://discord.gg/deepeval) 