from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.sdk.trace.export import (
    SpanExportResult,
    SpanExporter,
    ReadableSpan,
)
from pydantic import ValidationError, BaseModel
from typing import Any, Dict, List, Optional
from collections import defaultdict
import typing
import json

from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.types import (
    TraceSpanStatus,
    RetrieverSpan,
    AgentSpan,
    BaseSpan,
    LlmSpan,
    ToolSpan,
)
from deepeval.tracing.otel.utils import (
    check_tool_input_parameters_from_gen_ai_attributes,
    check_span_type_from_gen_ai_attributes,
    check_model_from_gen_ai_attributes,
    check_llm_input_from_gen_ai_attributes,
    check_tool_name_from_gen_ai_attributes,
    set_trace_time,
    to_hex_string,
    parse_string,
    parse_list_of_strings,
    post_test_run,
)
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.tracing.types import TraceAttributes
from deepeval.test_case import ToolCall
from dataclasses import dataclass
import deepeval


@dataclass
class BaseSpanWrapper:
    base_span: BaseSpan
    trace_input: Optional[Any] = None
    trace_output: Optional[Any] = None
    # trace attributes (to be deprecated)
    trace_attributes: Optional[TraceAttributes] = None
    # trace attributes
    trace_name: Optional[str] = None
    trace_tags: Optional[List[str]] = None
    trace_metadata: Optional[Dict[str, Any]] = None
    trace_thread_id: Optional[str] = None
    trace_user_id: Optional[str] = None
    trace_retrieval_context: Optional[List[str]] = None
    trace_context: Optional[List[str]] = None
    trace_tools_called: Optional[List[ToolCall]] = None
    trace_expected_tools: Optional[List[ToolCall]] = None
    trace_metric_collection: Optional[str] = None
    trace_environment: Optional[str] = None


class ConfidentSpanExporter(SpanExporter):

    def __init__(self, api_key: Optional[str] = None):
        capture_tracing_integration("otel.ConfidentSpanExporter")
        peb.init_clock_bridge()

        if api_key:
            deepeval.login(api_key)

        super().__init__()

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def export(
        self,
        spans: typing.Sequence[ReadableSpan],
        timeout_millis: int = 30000,
        api_key: Optional[str] = None,  # dynamic api key,
        _test_run_id: Optional[str] = None,
    ) -> SpanExportResult:
        # build forest of spans
        forest = self._build_span_forest(spans)

        # convert forest of spans to forest of base span wrappers
        spans_wrappers_forest: List[List[BaseSpanWrapper]] = []
        for span_list in forest:
            spans_wrappers_list: List[BaseSpanWrapper] = []
            for span in span_list:

                base_span_wrapper = self._convert_readable_span_to_base_span(
                    span
                )

                spans_wrappers_list.append(base_span_wrapper)
            spans_wrappers_forest.append(spans_wrappers_list)

        # add spans to trace manager
        for spans_wrappers_list in spans_wrappers_forest:
            for base_span_wrapper in spans_wrappers_list:

                current_trace = trace_manager.get_trace_by_uuid(
                    base_span_wrapper.base_span.trace_uuid
                )
                if not current_trace:
                    current_trace = trace_manager.start_new_trace(
                        trace_uuid=base_span_wrapper.base_span.trace_uuid
                    )

                if api_key:
                    current_trace.confident_api_key = api_key

                # error trace if root span is errored
                if base_span_wrapper.base_span.parent_uuid is None:
                    if (
                        base_span_wrapper.base_span.status
                        == TraceSpanStatus.ERRORED
                    ):
                        current_trace.status = TraceSpanStatus.ERRORED

                # set the trace attributes (to be deprecated)
                if base_span_wrapper.trace_attributes:

                    if base_span_wrapper.trace_attributes.name:
                        current_trace.name = (
                            base_span_wrapper.trace_attributes.name
                        )

                    if base_span_wrapper.trace_attributes.tags:
                        current_trace.tags = (
                            base_span_wrapper.trace_attributes.tags
                        )

                    if base_span_wrapper.trace_attributes.thread_id:
                        current_trace.thread_id = (
                            base_span_wrapper.trace_attributes.thread_id
                        )

                    if base_span_wrapper.trace_attributes.user_id:
                        current_trace.user_id = (
                            base_span_wrapper.trace_attributes.user_id
                        )

                    if base_span_wrapper.trace_attributes.metadata:
                        current_trace.metadata = (
                            base_span_wrapper.trace_attributes.metadata
                        )

                # set the trace attributes
                if base_span_wrapper.trace_name and isinstance(
                    base_span_wrapper.trace_name, str
                ):
                    current_trace.name = base_span_wrapper.trace_name

                if base_span_wrapper.trace_tags and isinstance(
                    base_span_wrapper.trace_tags, list
                ):
                    try:
                        current_trace.tags = [
                            str(tag) for tag in base_span_wrapper.trace_tags
                        ]
                    except Exception:
                        pass

                if base_span_wrapper.trace_metadata and isinstance(
                    base_span_wrapper.trace_metadata, dict
                ):
                    try:
                        current_trace.metadata = (
                            base_span_wrapper.trace_metadata
                        )
                    except Exception:
                        pass

                if base_span_wrapper.trace_thread_id and isinstance(
                    base_span_wrapper.trace_thread_id, str
                ):
                    current_trace.thread_id = base_span_wrapper.trace_thread_id

                if base_span_wrapper.trace_user_id and isinstance(
                    base_span_wrapper.trace_user_id, str
                ):
                    current_trace.user_id = base_span_wrapper.trace_user_id

                # set the trace input and output
                if base_span_wrapper.trace_input:
                    current_trace.input = base_span_wrapper.trace_input
                if base_span_wrapper.trace_output:
                    current_trace.output = base_span_wrapper.trace_output

                # set the trace environment
                if base_span_wrapper.trace_environment:
                    current_trace.environment = (
                        base_span_wrapper.trace_environment
                    )

                # set the trace test case parameters
                if base_span_wrapper.trace_retrieval_context:
                    current_trace.retrieval_context = (
                        base_span_wrapper.trace_retrieval_context
                    )
                if base_span_wrapper.trace_context:
                    current_trace.context = base_span_wrapper.trace_context
                if base_span_wrapper.trace_tools_called:
                    current_trace.tools_called = (
                        base_span_wrapper.trace_tools_called
                    )
                if base_span_wrapper.trace_expected_tools:
                    current_trace.expected_tools = (
                        base_span_wrapper.trace_expected_tools
                    )

                # set the trace metric collection
                if base_span_wrapper.trace_metric_collection:
                    current_trace.metric_collection = (
                        base_span_wrapper.trace_metric_collection
                    )

                trace_manager.add_span(base_span_wrapper.base_span)
                trace_manager.add_span_to_trace(base_span_wrapper.base_span)
                # no removing span because it can be parent of other spans

        # safely end all active traces or return them for test runs
        active_traces_keys = list(trace_manager.active_traces.keys())
        if _test_run_id:
            traces = []
            for trace_key in active_traces_keys:
                set_trace_time(trace_manager.get_trace_by_uuid(trace_key))
                trace = trace_manager.get_trace_by_uuid(trace_key)
                if trace:
                    traces.append(trace)
            trace_manager.clear_traces()
            post_test_run(traces, _test_run_id)
            return SpanExportResult.SUCCESS
        else:
            for trace_key in active_traces_keys:
                set_trace_time(trace_manager.get_trace_by_uuid(trace_key))
                trace_manager.end_trace(trace_key)
            trace_manager.clear_traces()
            return SpanExportResult.SUCCESS

    def _convert_readable_span_to_base_span(
        self, span: ReadableSpan
    ) -> BaseSpanWrapper:

        # Create typed spans
        base_span = None
        try:
            base_span = self._prepare_boilerplate_base_span(span)
        except Exception:
            pass

        # Creaete base span if no typed span
        parent_uuid = (
            to_hex_string(span.parent.span_id, 16) if span.parent else None
        )

        base_span_status = TraceSpanStatus.SUCCESS
        base_span_error = None

        if isinstance(span.status, Status):
            if span.status.status_code == StatusCode.ERROR:
                base_span_status = TraceSpanStatus.ERRORED
                base_span_error = span.status.description

        if not base_span:
            base_span = BaseSpan(
                uuid=to_hex_string(span.context.span_id, 16),
                status=base_span_status,
                children=[],
                trace_uuid=to_hex_string(span.context.trace_id, 32),
                parent_uuid=parent_uuid,
                start_time=peb.epoch_nanos_to_perf_seconds(span.start_time),
                end_time=peb.epoch_nanos_to_perf_seconds(span.end_time),
            )

        # Extract Span Attributes
        span_input = span.attributes.get("confident.span.input")
        span_output = span.attributes.get("confident.span.output")
        span_name = span.attributes.get("confident.span.name")

        raw_span_metric_collection = span.attributes.get(
            "confident.span.metric_collection"
        )
        raw_span_context = span.attributes.get("confident.span.context")
        raw_span_retrieval_context = span.attributes.get(
            "confident.span.retrieval_context"
        )
        raw_span_tools_called = span.attributes.get(
            "confident.span.tools_called"
        )
        if raw_span_tools_called and isinstance(raw_span_tools_called, tuple):
            raw_span_tools_called = list(raw_span_tools_called)

        raw_span_expected_tools = span.attributes.get(
            "confident.span.expected_tools"
        )
        if raw_span_expected_tools and isinstance(
            raw_span_expected_tools, tuple
        ):
            raw_span_expected_tools = list(raw_span_expected_tools)

        raw_span_metadata = span.attributes.get("confident.span.metadata")

        # Extract Trace Attributes
        trace_name = span.attributes.get("confident.trace.name")
        trace_thread_id = span.attributes.get("confident.trace.thread_id")
        trace_user_id = span.attributes.get("confident.trace.user_id")
        trace_environment = span.attributes.get(
            "confident.trace.environment", "production"
        )
        trace_input = span.attributes.get("confident.trace.input")
        trace_output = span.attributes.get("confident.trace.output")
        raw_trace_tags = span.attributes.get("confident.trace.tags")
        raw_trace_metadata = span.attributes.get("confident.trace.metadata")
        raw_trace_retrieval_context = span.attributes.get(
            "confident.trace.retrieval_context"
        )
        raw_trace_context = span.attributes.get("confident.trace.context")
        raw_trace_tools_called = span.attributes.get(
            "confident.trace.tools_called"
        )
        if raw_trace_tools_called and isinstance(raw_trace_tools_called, tuple):
            raw_trace_tools_called = list(raw_trace_tools_called)

        raw_trace_expected_tools = span.attributes.get(
            "confident.trace.expected_tools"
        )
        if raw_trace_expected_tools and isinstance(
            raw_trace_expected_tools, tuple
        ):
            raw_trace_expected_tools = list(raw_trace_expected_tools)

        raw_trace_metric_collection = span.attributes.get(
            "confident.trace.metric_collection"
        )

        # Validate Span Attributes
        span_retrieval_context = parse_list_of_strings(
            raw_span_retrieval_context
        )
        span_context = parse_list_of_strings(raw_span_context)
        span_tools_called = self._parse_list_of_tools(raw_span_tools_called)
        span_expected_tools = self._parse_list_of_tools(raw_span_expected_tools)
        span_metadata = self._parse_json_string(raw_span_metadata)
        span_metric_collection = parse_string(raw_span_metric_collection)

        # Validate Trace Attributes
        trace_tags = parse_list_of_strings(raw_trace_tags)
        trace_retrieval_context = parse_list_of_strings(
            raw_trace_retrieval_context
        )
        trace_context = parse_list_of_strings(raw_trace_context)
        trace_tools_called = self._parse_list_of_tools(raw_trace_tools_called)
        trace_expected_tools = self._parse_list_of_tools(
            raw_trace_expected_tools
        )
        trace_metadata = self._parse_json_string(raw_trace_metadata)
        trace_metric_collection = parse_string(raw_trace_metric_collection)

        # Set Span Attributes
        base_span.parent_uuid = (
            to_hex_string(span.parent.span_id, 16) if span.parent else None
        )
        base_span.name = None if base_span.name == "None" else base_span.name
        base_span.name = span_name or base_span.name or span.name
        base_span.status = base_span_status  # setting for boilerplate spans
        base_span.error = base_span_error
        if span_metric_collection:
            base_span.metric_collection = span_metric_collection
        if span_retrieval_context:
            base_span.retrieval_context = span_retrieval_context
        if span_context:
            base_span.context = span_context
        if span_tools_called:
            base_span.tools_called = span_tools_called
        if span_expected_tools:
            base_span.expected_tools = span_expected_tools
        if span_metadata:
            base_span.metadata = span_metadata
        if span_input:
            base_span.input = span_input
        if span_output:
            base_span.output = span_output

        # Resource attributes
        resource_attributes = span.resource.attributes
        if resource_attributes:
            environment = resource_attributes.get("confident.trace.environment")
            if environment and isinstance(environment, str):
                trace_environment = environment

        return BaseSpanWrapper(
            base_span=base_span,
            trace_input=trace_input,
            trace_output=trace_output,
            trace_name=trace_name,
            trace_tags=trace_tags,
            trace_metadata=trace_metadata,
            trace_thread_id=trace_thread_id,
            trace_user_id=trace_user_id,
            trace_retrieval_context=trace_retrieval_context,
            trace_context=trace_context,
            trace_tools_called=trace_tools_called,
            trace_expected_tools=trace_expected_tools,
            trace_metric_collection=trace_metric_collection,
            trace_environment=trace_environment,
        )

    def _prepare_boilerplate_base_span(
        self, span: ReadableSpan
    ) -> Optional[BaseSpan]:
        span_type = span.attributes.get("confident.span.type")
        if not span_type:
            span_type = check_span_type_from_gen_ai_attributes(span)

        # required fields
        uuid = to_hex_string(span.context.span_id, 16)
        status = (
            TraceSpanStatus.ERRORED
            if span.status.status_code == StatusCode.ERROR
            else TraceSpanStatus.SUCCESS
        )
        children = []
        trace_uuid = to_hex_string(span.context.trace_id, 32)
        parent_uuid = (
            to_hex_string(span.parent.span_id, 16) if span.parent else None
        )
        start_time = peb.epoch_nanos_to_perf_seconds(span.start_time)
        end_time = peb.epoch_nanos_to_perf_seconds(span.end_time)

        #######################################################
        ### LLM Span
        #######################################################

        if span_type == "llm":
            model = span.attributes.get("confident.llm.model")
            if not model:
                model = check_model_from_gen_ai_attributes(span)
            prompt = span.attributes.get("confident.llm.prompt")
            input_token_count = span.attributes.get(
                "confident.llm.input_token_count"
            )
            output_token_count = span.attributes.get(
                "confident.llm.output_token_count"
            )
            cost_per_input_token = span.attributes.get(
                "confident.llm.cost_per_input_token"
            )
            cost_per_output_token = span.attributes.get(
                "confident.llm.cost_per_output_token"
            )
            input, output = check_llm_input_from_gen_ai_attributes(span)
            if isinstance(input, tuple):
                input = list(input)
                try:
                    input = [json.loads(i) for i in input]
                except Exception:
                    pass
            if isinstance(output, tuple):
                output = list(output)
                try:
                    output = [json.loads(o) for o in output]
                except Exception:
                    pass
            llm_span = LlmSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,
                # llm span attributes
                model=model,
                cost_per_input_token=cost_per_input_token,
                cost_per_output_token=cost_per_output_token,
                prompt=prompt,
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                input=input,
                output=output,
            )
            return llm_span

        #######################################################
        ### Agent Span
        #######################################################

        elif span_type == "agent":
            name = span.attributes.get("confident.agent.name")
            available_tools_attr = span.attributes.get(
                "confident.agent.available_tools"
            )
            agent_handoffs_attr = span.attributes.get(
                "confident.agent.agent_handoffs"
            )
            available_tools: List[str] = []
            if available_tools_attr:
                try:
                    for tool in available_tools_attr:
                        available_tools.append(str(tool))
                except Exception:
                    pass
            agent_handoffs: List[str] = []
            if agent_handoffs_attr:
                try:
                    for handoff in agent_handoffs_attr:
                        agent_handoffs.append(str(handoff))
                except Exception:
                    pass
            agent_span = AgentSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,
                # agent span attributes
                name=name if name else "",
                available_tools=available_tools,
                agent_handoffs=agent_handoffs,
            )
            return agent_span

        #######################################################
        ### Retriever Span
        #######################################################

        elif span_type == "retriever":
            embedder = span.attributes.get("confident.retriever.embedder")
            top_k = span.attributes.get("confident.retriever.top_k")
            chunk_size = span.attributes.get("confident.retriever.chunk_size")
            retriever_span = RetrieverSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,
                # retriever span attributes
                embedder=embedder if embedder else "",
                top_k=top_k,
                chunk_size=chunk_size,
            )
            return retriever_span

        #######################################################
        ### Tool Span
        #######################################################

        elif span_type == "tool":
            name = span.attributes.get("confident.tool.name")
            if not name:
                name = check_tool_name_from_gen_ai_attributes(span)
            description = span.attributes.get("confident.tool.description")
            input = check_tool_input_parameters_from_gen_ai_attributes(span)

            tool_span = ToolSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,
                # tool span attributes
                name=name if name else "",
                description=description,
                input=input,
            )
            return tool_span

        return None

    #######################################################
    ### validation and Parsing
    #######################################################

    def _parse_base_model(
        self,
        base_model_json_str: str,
        base_model_type: BaseModel,
    ) -> Optional[BaseModel]:
        if base_model_json_str:
            try:
                return base_model_type.model_validate_json(base_model_json_str)
            except ValidationError:
                pass
        return None

    def _parse_json_string(self, json_str: str) -> Optional[Dict]:
        if json_str and isinstance(json_str, str):
            try:
                return json.loads(json_str)
            except Exception:
                pass
        return None

    def _parse_list_of_tools(self, tools: List[str]) -> List[ToolCall]:
        parsed_tools: List[ToolCall] = []
        if tools and isinstance(tools, list):
            for tool_json_str in tools:
                if isinstance(tool_json_str, str):
                    try:
                        parsed_tools.append(
                            ToolCall.model_validate_json(tool_json_str)
                        )
                    except ValidationError:
                        pass
        return parsed_tools

    #######################################################
    ### Span Forest
    #######################################################

    def _build_span_forest(
        self, spans: typing.Sequence[ReadableSpan]
    ) -> List[typing.Sequence[ReadableSpan]]:

        # Group spans by trace ID
        trace_spans = defaultdict(list)
        for span in spans:
            trace_id = span.context.trace_id
            trace_spans[trace_id].append(span)

        forest = []

        # Process each trace separately
        for trace_id, trace_span_list in trace_spans.items():
            # Build parent-child relationships for this trace
            children = defaultdict(list)
            span_map = {}
            all_span_ids = set()
            parent_map = {}

            for span in trace_span_list:
                span_id = span.context.span_id
                parent_id = span.parent.span_id if span.parent else None

                all_span_ids.add(span_id)
                span_map[span_id] = span
                parent_map[span_id] = parent_id

                if parent_id is not None:
                    children[parent_id].append(span_id)

            # Identify roots: spans with no parent or parent not in this trace
            roots = []
            for span_id in all_span_ids:
                parent_id = parent_map.get(span_id)
                if parent_id is None or parent_id not in all_span_ids:
                    roots.append(span_id)

            # Perform DFS from each root to collect spans in DFS order
            def dfs(start_id):
                order = []
                stack = [start_id]
                while stack:
                    current_id = stack.pop()
                    if current_id in span_map:  # Only add if span exists
                        order.append(span_map[current_id])
                    # Add children in reverse so that leftmost child is processed first
                    for child_id in sorted(children[current_id], reverse=True):
                        stack.append(child_id)
                return order

            # Build forest for this trace
            for root_id in sorted(roots):
                tree_order = dfs(root_id)
                if tree_order:  # Only add non-empty trees
                    forest.append(tree_order)

        return forest
