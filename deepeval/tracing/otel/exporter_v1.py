import json
import typing
from typing import List
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, ReadableSpan
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, TraceSpanStatus
from deepeval.tracing.otel.utils import to_hex_string
import deepeval
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
            #TODO: Fix timing of the trace 
            trace_manager.add_span(base_span)
            trace_manager.add_span_to_trace(base_span)
            # no removing span because it can be parent of other spans

        # safely end all active traces
        active_traces_keys = list(trace_manager.active_traces.keys())
        
        for trace_key in active_traces_keys:
            trace_manager.end_trace(trace_key)
        
        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
    
    def _convert_readable_span_to_base_span(self, span: ReadableSpan):

        return BaseSpan(
            name=span.name,
            uuid=to_hex_string(span.context.span_id, 16),
            parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
            trace_uuid=to_hex_string(span.context.trace_id, 32),
            start_time=span.start_time/1e9,
            end_time=span.end_time/1e9,
            status=TraceSpanStatus.SUCCESS, # TODO: handle status
            children=[],
            metadata=json.loads(span.to_json())
        )