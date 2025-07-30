import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from opentelemetry.sdk.trace.export import SpanExportResult, ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.span import Span

from deepeval.tracing.otel import ConfidentSpanExporter
from deepeval.tracing.types import BaseSpan, TraceSpanStatus
from deepeval.tracing.otel.utils import to_hex_string, set_trace_time


class TestConfidentSpanExporter(unittest.TestCase):
    pass