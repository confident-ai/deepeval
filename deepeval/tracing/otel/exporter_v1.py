from time import perf_counter
from datetime import datetime, timezone
import json
import typing
from typing import List, Optional
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, ReadableSpan
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, TraceSpanStatus
from deepeval.tracing.otel.utils import to_hex_string
from deepeval.tracing.tracing import to_zod_compatible_iso
from enum import Enum

from deepeval.integrations.pydantic_ai.handler import otel_span_handler as pydantic_ai_otel_span_handler

class FrameworkEnum(str, Enum):
    DEFAULT = "default"
    PYDANTIC_AI = "pydantic_ai"

class ConfidentSpanExporterV1(SpanExporter):
    active_trace_id: Optional[str] = None #TODO: introduce support for distributed systems
    framework: FrameworkEnum

    def __init__(self, framework: Optional[FrameworkEnum] = FrameworkEnum.DEFAULT):
        self.framework = framework
        capture_tracing_integration("deepeval.tracing.otel.exporter_v1")
        super().__init__()

    def check_active_trace_id(self):
        if self.active_trace_id is None:
            self.active_trace_id = trace_manager.start_new_trace().uuid
    
    def add_span_to_trace(self, span: BaseSpan):
        trace_manager.add_span(span)
        trace_manager.add_span_to_trace(span)
    
    def end_trace(self):
        trace_manager.end_trace(self.active_trace_id)
        self.active_trace_id = None
    
    def handle_default_framework_span(self, span: ReadableSpan):
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
    
    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        self.check_active_trace_id()
        spans_list: List[BaseSpan] = []
        for span in spans:
            if self.framework == FrameworkEnum.DEFAULT:
                confident_span = self.handle_default_framework_span(span)
            elif self.framework == FrameworkEnum.PYDANTIC_AI:
                confident_span = pydantic_ai_otel_span_handler(span, self.active_trace_id)
            else:
                ValueError(f"Framework {self.framework} not supported")
    
            spans_list.append(confident_span)
        
        for base_span in reversed(spans_list):
            self.add_span_to_trace(base_span)
        self.end_trace()

        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
    