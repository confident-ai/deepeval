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
            self.add_span_to_trace(base_span)
        
        self.end_trace()

        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
    
    
    def add_span_to_trace(self, span: BaseSpan):
        trace_manager.add_span(span)
        trace_manager.add_span_to_trace(span)
    
    def end_trace(self):
        trace_manager.end_trace(self.active_trace_id)
        self.active_trace_id = None
    
    def _convert_readable_span_to_base_span(self, span: ReadableSpan):
        return BaseSpan(
            uuid=to_hex_string(span.context.span_id, 16),
            status=TraceSpanStatus.SUCCESS, # TODO: handle status
            children=[],
            trace_uuid=self.active_trace_id,
            start_time=span.start_time/1e9,
            end_time=span.end_time/1e9,
            parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
            name=span.name,
            metadata=json.loads(span.to_json())
        )
    