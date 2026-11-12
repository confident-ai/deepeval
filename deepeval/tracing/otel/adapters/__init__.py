"""Adapters for frameworks whose OTel spans use their own vocabulary.

Like the platform's fallback integration hooks, an adapter fills only what
``confident.*`` attributes and the GenAI semantic conventions leave unset.
"""

import logging
from typing import List, Tuple

from opentelemetry.sdk.trace import ReadableSpan

from deepeval.tracing.otel.adapters.base import Adapter
from deepeval.tracing.otel.adapters.google_adk import GoogleADKAdapter
from deepeval.tracing.otel.adapters.llamaindex import LlamaIndexAdapter
from deepeval.tracing.otel.adapters.openinference import OpenInferenceAdapter
from deepeval.tracing.otel.adapters.pydantic_ai import PydanticAIAdapter
from deepeval.tracing.otel.adapters.strands import StrandsAdapter

logger = logging.getLogger(__name__)

ADAPTERS: Tuple[Adapter, ...] = (
    LlamaIndexAdapter(),
    PydanticAIAdapter(),
    StrandsAdapter(),
    GoogleADKAdapter(),
    # Framework adapters come first: this one reads any OpenInference span.
    OpenInferenceAdapter(),
)


def detect_adapters(span: ReadableSpan) -> List[Adapter]:
    detected = []
    for adapter in ADAPTERS:
        try:
            if adapter.detect(span):
                detected.append(adapter)
        except Exception:
            logger.debug("%s failed to inspect a span", type(adapter).__name__)
    return detected
