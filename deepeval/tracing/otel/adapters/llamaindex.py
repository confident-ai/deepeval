"""LlamaIndex, as confident-trace traces it.

Its retrievers run as generic ``Class.method`` spans, so the public
``retrieve`` / ``aretrieve`` call is what marks one as a retriever.
"""

from typing import Optional

from opentelemetry.sdk.trace import ReadableSpan

from deepeval.tracing.otel.adapters.base import Adapter
from deepeval.tracing.otel.attributes import ConfidentAttr

_INTEGRATION = "LlamaIndex"
_RETRIEVAL_METHODS = (".retrieve", ".aretrieve")


class LlamaIndexAdapter(Adapter):
    def detect(self, span: ReadableSpan) -> bool:
        attributes = span.attributes or {}
        return attributes.get(ConfidentAttr.SPAN_INTEGRATION) == _INTEGRATION

    def start_span_type(self, span: ReadableSpan) -> Optional[str]:
        return self.span_type(span)

    def span_type(self, span: ReadableSpan) -> Optional[str]:
        return "retriever" if span.name.endswith(_RETRIEVAL_METHODS) else None
