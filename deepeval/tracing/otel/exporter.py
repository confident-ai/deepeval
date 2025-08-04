import json
import os
import typing
from dataclasses import dataclass
from typing import List, Optional
from opentelemetry.trace.status import StatusCode
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
    ReadableSpan,
)
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.attributes import (
    LlmAttributes,
    RetrieverAttributes,
)
from deepeval.tracing.types import (
    AgentSpan,
    BaseSpan,
    LlmSpan,
    RetrieverSpan,
    ToolSpan,
    TraceSpanStatus,
)
from deepeval.tracing.otel.utils import (
    to_hex_string,
    set_trace_time,
    validate_llm_test_case_data,
    check_llm_input_from_gen_ai_attributes,
    check_tool_name_from_gen_ai_attributes,
    check_tool_input_parameters_from_gen_ai_attributes,
    check_span_type_from_gen_ai_attributes,
    check_model_from_gen_ai_attributes,
)
import deepeval
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.test_case import LLMTestCase, ToolCall
from pydantic import ValidationError
from deepeval.feedback.feedback import Feedback
from collections import defaultdict
from deepeval.tracing.attributes import TraceAttributes

@dataclass
class BaseSpanWrapper:
    base_span: BaseSpan
    trace_attributes: Optional[TraceAttributes] = None


class ConfidentSpanExporter(SpanExporter):

    def __init__(self, api_key: Optional[str] = None):
        capture_tracing_integration("deepeval.tracing.otel.exporter")
        peb.init_clock_bridge()

        if api_key:
            deepeval.login(api_key)

        environment = os.getenv("CONFIDENT_TRACE_ENVIRONMENT")
        if environment:
            trace_manager.configure(environment=environment)

        sampling_rate = os.getenv("CONFIDENT_SAMPLE_RATE")
        if sampling_rate:
            trace_manager.configure(sampling_rate=sampling_rate)

        super().__init__()

    def export(
        self,
        spans: typing.Sequence[ReadableSpan],
        timeout_millis: int = 30000,
        api_key: Optional[str] = None, # dynamic api key
    ) -> SpanExportResult:
        # build forest of spans
        forest = self._build_span_forest(spans)

        # convert forest of spans to forest of base span wrappers
        spans_wrappers_forest: List[List[BaseSpanWrapper]] = []
        for span_list in forest:
            spans_wrappers_list: List[BaseSpanWrapper] = []
            for span in span_list:

                # confugarion are attached to the resource attributes
                resource_attributes = span.resource.attributes
                environment = resource_attributes.get("confident.environment")

                if environment and isinstance(environment, str):
                    trace_manager.configure(environment=environment)

                sampling_rate = resource_attributes.get(
                    "confident.sampling_rate"
                )
                if sampling_rate and isinstance(sampling_rate, float):
                    trace_manager.configure(sampling_rate=sampling_rate)

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

                # set the trace attributes
                if base_span_wrapper.trace_attributes:
                    current_trace.name = base_span_wrapper.trace_attributes.name
                    current_trace.tags = base_span_wrapper.trace_attributes.tags
                    current_trace.thread_id = (
                        base_span_wrapper.trace_attributes.thread_id
                    )
                    current_trace.user_id = (
                        base_span_wrapper.trace_attributes.user_id
                    )

                trace_manager.add_span(base_span_wrapper.base_span)
                trace_manager.add_span_to_trace(base_span_wrapper.base_span)
                # no removing span because it can be parent of other spans

        # safely end all active traces
        active_traces_keys = list(trace_manager.active_traces.keys())
        for trace_key in active_traces_keys:
            set_trace_time(trace_manager.get_trace_by_uuid(trace_key))
            trace_manager.end_trace(trace_key)
        trace_manager.clear_traces()

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def _convert_readable_span_to_base_span(
        self, span: ReadableSpan
    ) -> BaseSpanWrapper:
        base_span = None
        try:
            base_span = self._prepare_boilerplate_base_span(span)
        except Exception as e:
            print(f"Error converting span: {e}")

        # fallback to base span with default values
        if not base_span:
            input = span.attributes.get("confident.span.input")
            output = span.attributes.get("confident.span.output")

            base_span = BaseSpan(
                uuid=to_hex_string(span.context.span_id, 16),
                status=(
                    TraceSpanStatus.ERRORED
                    if span.status.status_code == StatusCode.ERROR
                    else TraceSpanStatus.SUCCESS
                ),
                children=[],
                trace_uuid=to_hex_string(span.context.trace_id, 32),
                parent_uuid=(
                    to_hex_string(span.parent.span_id, 16)
                    if span.parent
                    else None
                ),
                start_time=peb.epoch_nanos_to_perf_seconds(span.start_time),
                end_time=peb.epoch_nanos_to_perf_seconds(span.end_time),
                input=input,
                output=output,
            )

        # extract error
        error = span.attributes.get("confident.span.error")

        # validate feedback
        feedback_json_str = span.attributes.get("confident.span.feedback")
        feedback = None
        if feedback_json_str:
            try:
                feedback = Feedback.model_validate_json(feedback_json_str)
            except ValidationError as err:
                print(f"Error converting feedback: {err}")

        # extract metric collection
        metric_collection = span.attributes.get(
            "confident.span.metric_collection"
        )
        if not isinstance(metric_collection, str):
            metric_collection = None

        # extract llm test case attributes (except additional_metadata, tools_called, expected_tools)
        test_case_input = span.attributes.get(
            "confident.span.llm_test_case.input"
        )
        test_case_actual_output = span.attributes.get(
            "confident.span.llm_test_case.actual_output"
        )
        test_case_expected_output = span.attributes.get(
            "confident.span.llm_test_case.expected_output"
        )
        test_case_context = span.attributes.get(
            "confident.span.llm_test_case.context"
        )
        test_case_retrieval_context = span.attributes.get(
            "confident.span.llm_test_case.retrieval_context"
        )

        # validate list of strings for tool calls and expected tools
        test_case_tools_called_attr = span.attributes.get(
            "confident.span.llm_test_case.tools_called"
        )
        test_case_expected_tools_attr = span.attributes.get(
            "confident.span.llm_test_case.expected_tools"
        )

        tools_called: List[ToolCall] = []
        expected_tools: List[ToolCall] = []

        if test_case_tools_called_attr and isinstance(
            test_case_tools_called_attr, list
        ):
            for tool_call_json_str in test_case_tools_called_attr:
                if isinstance(tool_call_json_str, str):
                    try:
                        tools_called.append(
                            ToolCall.model_validate_json(tool_call_json_str)
                        )
                    except ValidationError as err:
                        print(f"Error converting tool call: {err}")

        if test_case_expected_tools_attr and isinstance(
            test_case_expected_tools_attr, list
        ):
            for tool_call_json_str in test_case_expected_tools_attr:
                if isinstance(tool_call_json_str, str):
                    try:
                        expected_tools.append(
                            ToolCall.model_validate_json(tool_call_json_str)
                        )
                    except ValidationError as err:
                        print(f"Error converting expected tool call: {err}")

        llm_test_case = None
        if test_case_input and test_case_actual_output:
            try:
                validate_llm_test_case_data(
                    input=test_case_input,
                    actual_output=test_case_actual_output,
                    expected_output=test_case_expected_output,
                    context=test_case_context,
                    retrieval_context=test_case_retrieval_context,
                )

                llm_test_case = LLMTestCase(
                    input=test_case_input,
                    actual_output=test_case_actual_output,
                    expected_output=test_case_expected_output,
                    context=test_case_context,
                    retrieval_context=test_case_retrieval_context,
                    tools_called=tools_called,
                    expected_tools=expected_tools,
                )
            except Exception as e:
                print(f"Invalid LLMTestCase data: {e}")

        base_span.parent_uuid = (to_hex_string(span.parent.span_id, 16) if span.parent else None)
        
        # base span name takes precedence over span name
        _name = None
        if base_span.name is not None and base_span.name != "None":
            _name = base_span.name

        base_span.name = _name if _name else span.name
        base_span.metadata = json.loads(span.to_json())
        base_span.error = error
        base_span.llm_test_case = llm_test_case
        base_span.metric_collection = metric_collection
        base_span.feedback = feedback

        # extract trace attributes
        trace_attributes = None
        trace_attr_json_str = span.attributes.get("confident.trace.attributes")
        if trace_attr_json_str:
            try:
                trace_attributes = TraceAttributes.model_validate_json(
                    trace_attr_json_str
                )
            except ValidationError as err:
                print(f"Error converting trace attributes: {err}")

        base_span_wrapper = BaseSpanWrapper(
            base_span=base_span, trace_attributes=trace_attributes
        )

        return base_span_wrapper

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

        if span_type == "llm":
            model = span.attributes.get("confident.llm.model")
            if not model:
                model = check_model_from_gen_ai_attributes(span)
            
            cost_per_input_token = span.attributes.get(
                "confident.llm.cost_per_input_token"
            )
            cost_per_output_token = span.attributes.get(
                "confident.llm.cost_per_output_token"
            )

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
            )

            # set attributes
            input = span.attributes.get("confident.llm.attributes.input")
            output = span.attributes.get("confident.llm.attributes.output")
            prompt = span.attributes.get("confident.llm.attributes.prompt")
            input_token_count = span.attributes.get(
                "confident.llm.attributes.input_token_count"
            )
            output_token_count = span.attributes.get(
                "confident.llm.attributes.output_token_count"
            )

            if not input and not output:
                input, output = check_llm_input_from_gen_ai_attributes(span)

            try:
                llm_span.set_attributes(
                    LlmAttributes(
                        input=input,
                        output=output,
                        prompt=prompt,
                        input_token_count=input_token_count,
                        output_token_count=output_token_count,
                    )
                )
            except Exception as e:
                print(f"Error setting llm span attributes: {e}")

            return llm_span

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
                except Exception as e:
                    print(f"Error converting available tools: {e}")

            agent_handoffs: List[str] = []
            if agent_handoffs_attr:
                try:
                    for handoff in agent_handoffs_attr:
                        agent_handoffs.append(str(handoff))
                except Exception as e:
                    print(f"Error converting agent handoffs: {e}")

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

            # set attributes
            input = span.attributes.get("confident.agent.attributes.input")
            output = span.attributes.get("confident.agent.attributes.output")

            try:
                agent_span.input = trace_manager.mask(input)
                agent_span.output = trace_manager.mask(output)
            except Exception as e:
                print(f"Error setting agent span attributes: {e}")

            return agent_span

        elif span_type == "retriever":
            embedder = span.attributes.get("confident.retriever.embedder")

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
            )

            # set attributes
            embedding_input = span.attributes.get(
                "confident.retriever.attributes.embedding_input"
            )
            retrieval_context = span.attributes.get(
                "confident.retriever.attributes.retrieval_context"
            )
            top_k = span.attributes.get("confident.retriever.attributes.top_k")
            chunk_size = span.attributes.get(
                "confident.retriever.attributes.chunk_size"
            )

            try:
                retriever_span.set_attributes(
                    RetrieverAttributes(
                        embedding_input=embedding_input,
                        retrieval_context=retrieval_context,
                        top_k=top_k,
                        chunk_size=chunk_size,
                    )
                )
            except Exception as e:
                print(f"Error setting retriever span attributes: {e}")

            return retriever_span

        elif span_type == "tool":
            name = span.attributes.get("confident.tool.name")
            if not name:
                name = check_tool_name_from_gen_ai_attributes(span)

            description = span.attributes.get("confident.tool.description")

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
            )

            # set attributes
            input_parameters = span.attributes.get("confident.tool.attributes.input_parameters")
            output = span.attributes.get("confident.tool.attributes.output")

            try:
                input_parameters = json.loads(input_parameters) if input_parameters else None
            except Exception as e:
                input_parameters = None
                print(f"Error converting input parameters: {e}")

            if not input_parameters:
                input_parameters = check_tool_input_parameters_from_gen_ai_attributes(span)

            try:
                tool_span.input = trace_manager.mask(input_parameters)
                tool_span.output = trace_manager.mask(output)
            except Exception as e:
                print(f"Error setting tool span attributes: {e}")

            return tool_span

        # if span type is not supported, return None
        return None

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
