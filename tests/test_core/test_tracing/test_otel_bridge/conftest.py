from types import SimpleNamespace
from unittest.mock import MagicMock
import importlib

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from deepeval.tracing.context import current_trace_context, current_span_context
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import EvalSession
from deepeval.tracing.otel.capture import finish_capture
from deepeval.tracing.otel.context_aware_processor import (
    ContextAwareSpanProcessor,
)
from deepeval.tracing.otel.provider import attach


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    trace_manager.clear_traces()
    trace_manager.eval_session = EvalSession()
    a = current_trace_context.set(None)
    b = current_span_context.set(None)
    monkeypatch.setattr(trace_manager, "post_trace", MagicMock())
    yield
    finish_capture(trace_manager.eval_session)
    trace_manager.clear_traces()
    trace_manager.eval_session = EvalSession()
    current_span_context.reset(b)
    current_trace_context.reset(a)


@pytest.fixture
def pipeline(monkeypatch, request):
    import deepeval.tracing.otel.context_aware_processor as module

    monkeypatch.setattr(module, "OTLPSpanExporter", lambda **kw: MagicMock())
    monkeypatch.setattr(
        module, "BatchSpanProcessor", lambda exporter: MagicMock()
    )
    integration = getattr(request, "param", "strands")
    classes = {
        "strands": "StrandsSpanInterceptor",
        "agentcore": "AgentCoreSpanInterceptor",
        "openinference": "OpenInferenceSpanInterceptor",
        "pydantic_ai": "SpanInterceptor",
    }
    module = importlib.import_module(
        f"deepeval.integrations.{integration}.instrumentator"
    )
    settings = SimpleNamespace(
        name=None,
        thread_id=None,
        user_id=None,
        metadata=None,
        tags=None,
        metric_collection=None,
        test_case_id=None,
        turn_id=None,
        environment="development",
        integration="Google ADK",
    )
    provider = TracerProvider(sampler=ALWAYS_ON, shutdown_on_exit=False)
    processor = ContextAwareSpanProcessor()
    interceptor = getattr(module, classes[integration])(settings)
    router = attach(provider, integration, interceptor, processor)
    yield (
        provider.get_tracer(
            {
                "strands": "strands.telemetry.tracer",
                "agentcore": "agentcore",
                "openinference": "openinference.instrumentation.google_adk",
                "pydantic_ai": "pydantic-ai",
            }[integration]
        ),
        processor,
        router,
        provider,
    )
    provider.shutdown()
