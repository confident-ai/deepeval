from typing import List, Dict, Any, Sequence
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
import json
from datetime import datetime


class TestExporter(SpanExporter):
    """This exporter is used to test the exporter. It will store the spans in a list of dictionaries."""

    span_json_list: List[Dict[str, Any]] = []

    def export(
        self, spans: Sequence[ReadableSpan], timeout_millis: int = 30000
    ) -> SpanExportResult:
        for span in spans:
            _span_json = json.loads(span.to_json())
            self.span_json_list.append(_span_json)

        return SpanExportResult.SUCCESS

    def get_span_json_list(self) -> List[Dict[str, Any]]:
        return sorted(
            self.span_json_list,
            key=lambda x: datetime.fromisoformat(
                x["start_time"].replace("Z", "+00:00")
            ),
        )

    def clear_span_json_list(self):
        self.span_json_list = []


test_exporter = TestExporter()
