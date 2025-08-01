from typing import Any, Optional, List, Dict
from uuid import UUID
from time import perf_counter
import uuid

from deepeval.tracing.attributes import (
    LlmAttributes,
    RetrieverAttributes,
    LlmOutput,
    LlmToolCall,
)
from deepeval.metrics import BaseMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import global_test_run_manager
from deepeval.evaluate.utils import create_api_test_case
from deepeval.test_run import LLMApiTestCase

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.outputs import ChatGeneration
    from langchain_core.messages import AIMessage

    # contains langchain imports
    from deepeval.integrations.langchain.utils import (
        parse_prompts_to_messages,
        prepare_dict,
        extract_token_usage,
        extract_name,
    )

    langchain_installed = True
except:
    langchain_installed = False


def is_langchain_installed():
    if not langchain_installed:
        raise ImportError(
            "LangChain is not installed. Please install it with `pip install langchain`."
        )


from deepeval.tracing import trace_manager
from deepeval.tracing.types import (
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    TraceSpanStatus,
    ToolSpan,
)
from deepeval.telemetry import capture_tracing_integration


class LangGraphCallbackHandler(BaseCallbackHandler):
    """
    Enhanced callback handler specifically designed for LangGraph applications.
    Provides detailed tracing for graph execution, node transitions, and state management.
    """

    active_trace_id: Optional[str] = None
    metrics: List[BaseMetric] = []
    metric_collection: Optional[str] = None
    graph_execution_id: Optional[str] = None
    current_node: Optional[str] = None
    node_execution_order: int = 0

    def __init__(
        self,
        metrics: List[BaseMetric] = [],
        metric_collection: Optional[str] = None,
        enable_graph_tracing: bool = True,
        enable_node_tracing: bool = True,
        enable_state_tracing: bool = True,
    ):
        capture_tracing_integration(
            "deepeval.integrations.langgraph.callback.LangGraphCallbackHandler"
        )
        is_langchain_installed()
        self.metrics = metrics
        self.metric_collection = metric_collection
        self.enable_graph_tracing = enable_graph_tracing
        self.enable_node_tracing = enable_node_tracing
        self.enable_state_tracing = enable_state_tracing
        super().__init__()

    def check_active_trace_id(self):
        if self.active_trace_id is None:
            self.active_trace_id = trace_manager.start_new_trace().uuid

    def add_span_to_trace(self, span: BaseSpan):
        trace_manager.add_span(span)
        trace_manager.add_span_to_trace(span)

    def end_span(self, span: BaseSpan):
        span.end_time = perf_counter()
        span.status = TraceSpanStatus.SUCCESS
        trace_manager.remove_span(str(span.uuid))

        # Add metric collection to root spans
        if self.metric_collection and span.parent_uuid is None:
            span.metric_collection = self.metric_collection

        # Add metrics to root spans
        if self.metrics and span.parent_uuid is None:
            span.metrics = self.metrics

            # Prepare test case for task completion metric
            for metric in self.metrics:
                if isinstance(metric, TaskCompletionMetric):
                    self.prepare_task_completion_test_case(span)

    def end_trace(self, span: BaseSpan):
        current_trace = trace_manager.get_trace_by_uuid(self.active_trace_id)

        # Send trace for evaluation if metrics are provided
        if self.metrics:
            trace_manager.evaluating = True
            trace_manager.evaluation_loop = True
            trace_manager.integration_traces_to_evaluate.append(current_trace)

        if current_trace is not None:
            current_trace.input = span.input
            current_trace.output = span.output
        trace_manager.end_trace(self.active_trace_id)
        self.active_trace_id = None
        self.graph_execution_id = None
        self.current_node = None
        self.node_execution_order = 0

    def prepare_task_completion_test_case(self, span: BaseSpan):
        test_case = LLMTestCase(input="None", actual_output="None")
        test_case._trace_dict = trace_manager.create_nested_spans_dict(span)
        span.llm_test_case = test_case

    # LangGraph-specific callback methods
    def on_graph_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Track the start of a LangGraph execution"""
        if not self.enable_graph_tracing:
            return

        self.check_active_trace_id()
        self.graph_execution_id = str(run_id)
        
        # Create graph execution span
        graph_span = BaseSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=serialized.get('name', 'LangGraph Execution'),
            input=inputs,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
        )
        
        # Add graph-specific metadata
        graph_span.metadata = graph_span.metadata or {}
        graph_span.metadata.update({
            'span_type': 'graph_execution',
            'graph_config': serialized.get('config', {}),
            'execution_mode': metadata.get('execution_mode', 'sequential') if metadata else 'sequential',
            'node_count': metadata.get('node_count', 0) if metadata else 0,
        })
        
        self.add_span_to_trace(graph_span)

    def on_graph_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Track the end of a LangGraph execution"""
        if not self.enable_graph_tracing:
            return

        graph_span = trace_manager.get_span_by_uuid(str(run_id))
        if graph_span is None:
            return

        graph_span.output = outputs
        self.end_span(graph_span)

        if parent_run_id is None:
            self.end_trace(graph_span)

    def on_node_start(
        self,
        node_name: str,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Track the start of a node execution"""
        if not self.enable_node_tracing:
            return

        self.check_active_trace_id()
        self.current_node = node_name
        self.node_execution_order += 1
        
        # Create node execution span
        node_span = BaseSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else self.graph_execution_id,
            start_time=perf_counter(),
            name=node_name,
            input=inputs,
            metadata=prepare_dict(
                serialized={'name': node_name}, tags=tags, metadata=metadata, **kwargs
            ),
        )
        
        # Add node-specific metadata
        node_span.metadata = node_span.metadata or {}
        node_span.metadata.update({
            'span_type': 'node_execution',
            'node_type': metadata.get('node_type', 'function') if metadata else 'function',
            'execution_order': self.node_execution_order,
            'dependencies': metadata.get('dependencies', []) if metadata else [],
            'conditional_logic': metadata.get('conditional_logic') if metadata else None,
            'parallel_group': metadata.get('parallel_group') if metadata else None,
        })
        
        self.add_span_to_trace(node_span)

    def on_node_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Track the end of a node execution"""
        if not self.enable_node_tracing:
            return

        node_span = trace_manager.get_span_by_uuid(str(run_id))
        if node_span is None:
            return

        node_span.output = outputs
        self.end_span(node_span)

    def on_state_transition(
        self,
        from_node: str,
        to_node: str,
        state: dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Track state transitions between nodes"""
        if not self.enable_state_tracing:
            return

        self.check_active_trace_id()
        
        # Create state transition span
        transition_span = BaseSpan(
            uuid=str(uuid.uuid4()),
            status=TraceSpanStatus.SUCCESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=self.graph_execution_id,
            start_time=perf_counter(),
            name=f"State Transition: {from_node} → {to_node}",
            input={'from_node': from_node, 'to_node': to_node, 'state': state},
            metadata={
                'span_type': 'state_transition',
                'from_node': from_node,
                'to_node': to_node,
                'state_changes': kwargs.get('state_changes', {}),
                'transition_condition': kwargs.get('transition_condition'),
                'routing_decision': kwargs.get('routing_decision'),
            },
        )
        
        self.add_span_to_trace(transition_span)
        self.end_span(transition_span)

    # Inherit LangChain callback methods for compatibility
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle chain start - treat as node execution if in LangGraph context"""
        if self.graph_execution_id:
            # If we're in a LangGraph context, treat this as a node
            node_name = extract_name(serialized, **kwargs)
            self.on_node_start(node_name, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
        else:
            # Fall back to standard chain handling
            self.check_active_trace_id()
            base_span = BaseSpan(
                uuid=str(run_id),
                status=TraceSpanStatus.IN_PROGRESS,
                children=[],
                trace_uuid=self.active_trace_id,
                parent_uuid=str(parent_run_id) if parent_run_id else None,
                start_time=perf_counter(),
                name=extract_name(serialized, **kwargs),
                input=inputs,
                metadata=prepare_dict(
                    serialized=serialized, tags=tags, metadata=metadata, **kwargs
                ),
            )
            self.add_span_to_trace(base_span)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle chain end - treat as node execution if in LangGraph context"""
        if self.graph_execution_id:
            # If we're in a LangGraph context, treat this as a node
            self.on_node_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        else:
            # Fall back to standard chain handling
            base_span = trace_manager.get_span_by_uuid(str(run_id))
            if base_span is None:
                return

            base_span.output = outputs
            self.end_span(base_span)

            if parent_run_id is None:
                self.end_trace(base_span)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle LLM start"""
        self.check_active_trace_id()

        # Extract input
        input_messages = parse_prompts_to_messages(prompts, **kwargs)

        llm_span = LlmSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=input_messages,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
        )

        # Set model information
        llm_span.model = serialized.get("name", "unknown")
        
        self.add_span_to_trace(llm_span)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle LLM end"""
        llm_span = trace_manager.get_span_by_uuid(str(run_id))
        if llm_span is None:
            return

        # Extract output
        if hasattr(response, "generations") and response.generations:
            generations = response.generations[0]
            if generations and hasattr(generations[0], "text"):
                output = generations[0].text
            elif generations and hasattr(generations[0], "message"):
                output = generations[0].message.content
            else:
                output = str(response)
        else:
            output = str(response)

        llm_span.output = output

        # Set LLM attributes
        llm_attributes = LlmAttributes(
            input=llm_span.input,
            output=output,
        )
        llm_span.set_attributes(llm_attributes)

        self.end_span(llm_span)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle tool start"""
        self.check_active_trace_id()

        tool_span = ToolSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=input_str,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
        )

        self.add_span_to_trace(tool_span)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle tool end"""
        tool_span = trace_manager.get_span_by_uuid(str(run_id))
        if tool_span is None:
            return

        tool_span.output = output

        # Set tool attributes
        tool_attributes = ToolAttributes(
            input_parameters={"input": tool_span.input},
            output=output,
        )
        tool_span.set_attributes(tool_attributes)

        self.end_span(tool_span)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle retriever start"""
        self.check_active_trace_id()

        retriever_span = RetrieverSpan(
            uuid=str(run_id),
            status=TraceSpanStatus.IN_PROGRESS,
            children=[],
            trace_uuid=self.active_trace_id,
            parent_uuid=str(parent_run_id) if parent_run_id else None,
            start_time=perf_counter(),
            name=extract_name(serialized, **kwargs),
            input=query,
            metadata=prepare_dict(
                serialized=serialized, tags=tags, metadata=metadata, **kwargs
            ),
            embedder=serialized.get("name", "unknown"),
        )

        self.add_span_to_trace(retriever_span)

    def on_retriever_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle retriever end"""
        retriever_span = trace_manager.get_span_by_uuid(str(run_id))
        if retriever_span is None:
            return

        retriever_span.output = output

        # Set retriever attributes
        retriever_attributes = RetrieverAttributes(
            embedding_input=retriever_span.input,
            retrieval_context=output,
        )
        retriever_span.set_attributes(retriever_attributes)

        self.end_span(retriever_span) 