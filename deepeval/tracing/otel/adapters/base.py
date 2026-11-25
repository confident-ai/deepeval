"""The contract every framework adapter implements."""

from typing import Any, Dict, FrozenSet, Optional

from opentelemetry.sdk.trace import ReadableSpan

from deepeval.tracing.types import SpanType


class Adapter:
    """One framework's mapping. Every hook may decline with an empty answer."""

    # Stamped on detected spans that confident-trace didn't label.
    integration: Optional[str] = None
    # Names the framework reports as the provider in place of the vendor.
    self_providers: FrozenSet[str] = frozenset()

    def detect(self, span: ReadableSpan) -> bool:
        return False

    def start_span_type(self, span: ReadableSpan) -> Optional[str]:
        """The span's type, as a wire name, from what is set when it starts."""
        return None

    def span_type(self, span: ReadableSpan) -> Optional[str]:
        """The span's type, as a wire name such as ``"llm"`` or ``"base"``."""
        return None

    def span_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        """deepeval span fields by name; only unset fields take them."""
        return {}

    def trace_fields(
        self, span: ReadableSpan, span_type: Optional[SpanType]
    ) -> Dict[str, Any]:
        """``Trace`` fields by name; only unset fields take them."""
        return {}

    def conversation_id(self, span: ReadableSpan) -> Optional[str]:
        return None
