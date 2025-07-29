import json
import typing
from typing import List, Optional
from opentelemetry.trace.status import StatusCode
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, ReadableSpan
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, TraceSpanStatus
from deepeval.tracing.otel.utils import to_hex_string, set_trace_time, validate_llm_test_case_data
import deepeval
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.test_case import LLMTestCase, ToolCall
from pydantic import ValidationError

class ConfidentSpanExporterV1(SpanExporter):
    
    def __init__(self, api_key: Optional[str] = None):
        capture_tracing_integration("deepeval.tracing.otel.exporter_v1")

        if api_key:
            deepeval.login_with_confident_api_key(api_key) # TODO: send api keys dynamically to get it compatible with the collector framework
        
        super().__init__()

    def export(
            self,
            spans: typing.Sequence[ReadableSpan],
            timeout_millis: int = 30000
        ) -> SpanExportResult:
        
        spans_list: List[BaseSpan] = []
        
        for span in spans:
            spans_list.append(self._convert_readable_span_to_base_span(span))
        
        # list starts from root span
        for base_span in reversed(spans_list):
            if not trace_manager.get_trace_by_uuid(base_span.trace_uuid):
                trace_manager.start_new_trace(trace_uuid=base_span.trace_uuid)
            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)
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
    
    def _convert_readable_span_to_base_span(self, span: ReadableSpan) -> BaseSpan:

        # extract metric collection
        metric_collection = span.attributes.get("confident.metric_collection")
        if not isinstance(metric_collection, str):
            metric_collection = None
        
        # extract llm test case attributes (except additional_metadata, tools_called, expected_tools)
        input = span.attributes.get("confident.llm_test_case.input")
        actual_output = span.attributes.get("confident.llm_test_case.actual_output")
        expected_output = span.attributes.get("confident.llm_test_case.expected_output")
        context = span.attributes.get("confident.llm_test_case.context")
        retrieval_context = span.attributes.get("confident.llm_test_case.retrieval_context")

        # validate list of strings for tool calls and expected tools
        tools_called_attr = span.attributes.get("confident.llm_test_case.tools_called")
        expected_tools_attr = span.attributes.get("confident.llm_test_case.expected_tools")

        tools_called: List[ToolCall] = []
        expected_tools: List[ToolCall] = []

        if tools_called_attr and isinstance(tools_called_attr, list):
            for tool_call_json_str in tools_called_attr:
                if isinstance(tool_call_json_str, str):
                    try:
                        tools_called.append(ToolCall.model_validate_json(tool_call_json_str))
                    except ValidationError as err:
                        print(f"Error converting tool call: {err}")
        
        if expected_tools_attr and isinstance(expected_tools_attr, list):
            for tool_call_json_str in expected_tools_attr:
                if isinstance(tool_call_json_str, str):
                    try:
                        expected_tools.append(ToolCall.model_validate_json(tool_call_json_str))
                    except ValidationError as err:
                        print(f"Error converting expected tool call: {err}")
        
        llm_test_case = None
        if input and actual_output:
            try:
                validate_llm_test_case_data(input, actual_output, expected_output, context, retrieval_context, tools_called, expected_tools)
                
                llm_test_case = LLMTestCase(
                    input=input,
                    actual_output=actual_output,
                    expected_output=expected_output,
                    context=context,
                    retrieval_context=retrieval_context,
                    tools_called=tools_called,
                    expected_tools=expected_tools
                )
            except Exception as e:
                print(f"Invalid LLMTestCase data: {e}")
        
        return BaseSpan(
            name=span.name,
            uuid=to_hex_string(span.context.span_id, 16),
            parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
            trace_uuid=to_hex_string(span.context.trace_id, 32),
            start_time=peb.epoch_nanos_to_perf_seconds(span.start_time),
            end_time=peb.epoch_nanos_to_perf_seconds(span.end_time),
            status=TraceSpanStatus.ERRORED if span.status.status_code == StatusCode.ERROR else TraceSpanStatus.SUCCESS,
            children=[],
            metadata=json.loads(span.to_json()),
            metric_collection=metric_collection,
            llm_test_case=llm_test_case
        )