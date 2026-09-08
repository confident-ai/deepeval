"""Run with the local confident-trace source on PYTHONPATH until publication."""

import json

import pytest


def test_actual_confident_trace_emission_enters_deepeval(pipeline):
    confident_trace = pytest.importorskip("confident_trace")
    from confident_trace._core.attachment import NativeAttachment
    from deepeval.tracing.context import current_trace_context
    from deepeval.tracing.tracing import Observer
    from deepeval.tracing.types import ToolSpan
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _, processor, router, provider = pipeline
    assert isinstance(router.native_attachment, NativeAttachment)
    # Public init still owns its exporter. A local memory exporter demonstrates
    # that attaching DeepEval does not replace or shut down that application sink.
    application_exporter = InMemorySpanExporter()
    confident_trace.init(
        tracer_provider=provider,
        exporter=application_exporter,
        instrumentations=(),
    )
    try:

        @confident_trace.span(name="lookup", kind="tool")
        def lookup(query):
            return {"answer": query}

        with Observer("agent", "outer"):
            captured = current_trace_context.get()
            assert lookup("local source") == {"answer": "local source"}
        confident_trace.flush()
        child = captured.root_spans[0].children[0]
        assert isinstance(child, ToolSpan)
        assert child.output == {"answer": "local source"}
        assert len(application_exporter.get_finished_spans()) == 1
        processor._otlp_processor.on_end.assert_not_called()
    finally:
        confident_trace.shutdown()


def test_local_confident_receiver_vectors():
    confident_trace = pytest.importorskip("confident_trace")
    from pathlib import Path
    from types import SimpleNamespace
    from opentelemetry.sdk.trace import Event
    from deepeval.tracing.otel.utils import (
        check_llm_input_from_gen_ai_attributes,
    )

    source = (
        Path(confident_trace.__file__).resolve().parents[3]
        / "spec/genai-vectors.json"
    )
    if not source.is_file():
        pytest.skip(
            "confident-trace source vectors are not installed with the wheel"
        )
    cases = json.loads(source.read_text())["cases"]
    for case in cases:
        span = SimpleNamespace(
            attributes=case["attributes"],
            events=[Event(e["name"], e["attributes"]) for e in case["events"]],
        )
        actual = check_llm_input_from_gen_ai_attributes(span)
        assert actual == (
            case["expected"]["input"],
            case["expected"]["output"],
        ), case["name"]
