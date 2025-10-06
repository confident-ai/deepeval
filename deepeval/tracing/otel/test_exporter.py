from typing import List, Dict, Any, Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

class TestExporter(SpanExporter):
    """This exporter is used to test the exporter. It will store the spans in a list of dictionaries."""
    span_json_list: List[Dict[str, Any]] = []

    def export(self, spans: Sequence[ReadableSpan], timeout_millis: int = 30000) -> SpanExportResult:
        for span in spans:
            self.span_json_list.append(span.to_json())
        return SpanExportResult.SUCCESS
    
    def get_span_json_list(self) -> List[Dict[str, Any]]:
        return self.span_json_list

test_exporter = TestExporter()