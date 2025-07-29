import json
import typing
from typing import List
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, ReadableSpan
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, TraceSpanStatus
from deepeval.tracing.otel.utils import to_hex_string, set_trace_time
import deepeval
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.test_case import LLMTestCase

class ConfidentSpanExporterV1(SpanExporter):
    
    def __init__(self, api_key: str):
        capture_tracing_integration("deepeval.tracing.otel.exporter_v1")

        deepeval.login_with_confident_api_key(api_key)
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

        metric_collection = span.attributes.get("confident.metric_collection", None)
        if not isinstance(metric_collection, str):
            metric_collection = None
        
        llm_test_case = None
        try:
            # Only create LLMTestCase if we have valid input and actual_output
            input_value = span.attributes.get("confident.llm_test_case.input", None)
            actual_output_value = span.attributes.get("confident.llm_test_case.actual_output", None)
            
            if input_value is not None and actual_output_value is not None:
                llm_test_case = LLMTestCase(
                    input=input_value,
                    actual_output=actual_output_value,
                    expected_output=span.attributes.get("confident.llm_test_case.expected_output", None),
                    context=span.attributes.get("confident.llm_test_case.context", None),
                    retrieval_context=span.attributes.get("confident.llm_test_case.retrieval_context", None),
                    additional_metadata=span.attributes.get("confident.llm_test_case.additional_metadata", None),
                    # tools_called=span.attributes.get("confident_ai.llm_test_case.tools_called", None), # TODO: handle tools_called
                    # expected_tools=span.attributes.get("confident_ai.llm_test_case.expected_tools", None), # TODO: handle expected_tools
                )
        except Exception as e:
            print(f"Error converting llm_test_case: {e}")
        
        return BaseSpan(
            name=span.name,
            uuid=to_hex_string(span.context.span_id, 16),
            parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
            trace_uuid=to_hex_string(span.context.trace_id, 32),
            start_time=peb.epoch_nanos_to_perf_seconds(span.start_time),
            end_time=peb.epoch_nanos_to_perf_seconds(span.end_time),
            status=TraceSpanStatus.SUCCESS, # TODO: handle status
            children=[],
            metadata=json.loads(span.to_json()),
            metric_collection=metric_collection,
            llm_test_case=llm_test_case
        )