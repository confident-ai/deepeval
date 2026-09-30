"""Deterministic bridge tests. No collector, provider credentials, or judge LLM."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import Event
from deepeval.contextvars import set_current_golden, reset_current_golden
from deepeval.dataset import Golden
from deepeval.tracing.context import (
    current_trace_context,
    update_current_span,
)
from deepeval.tracing.tracing import trace_manager, Observer
from deepeval.tracing.types import (
    EvalMode,
    EvalSession,
    LlmSpan,
    ToolSpan,
    TraceSpanStatus,
)
from deepeval.tracing.otel.capture import finish_capture
from deepeval.tracing.otel.utils import check_llm_input_from_gen_ai_attributes


def test_children_finish_before_parent_and_metrics_stay_local(pipeline):
    tracer, processor, _, _ = pipeline
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    golden = Golden(input="one")
    token = set_current_golden(golden)
    try:
        with tracer.start_as_current_span(
            "agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "agent",
            },
        ):
            with tracer.start_as_current_span(
                "tool",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup",
                },
            ):
                update_current_span(input={"query": "one"}, output="answer")
            assert not trace_manager.eval_session.traces_to_evaluate
    finally:
        reset_current_golden(token)
    traces = trace_manager.eval_session.traces_to_evaluate
    assert len(traces) == 1
    assert (
        trace_manager.eval_session.trace_uuid_to_golden[traces[0].uuid]
        is golden
    )
    assert traces[0].root_spans[0].children[0].input == {"query": "one"}
    assert isinstance(traces[0].root_spans[0].children[0], ToolSpan)
    assert not trace_manager.active_spans
    assert not processor._capture.bindings
    processor._otlp_processor.on_end.assert_not_called()
    trace_manager.post_trace.assert_not_called()
    assert current_trace_context.get() is None


def test_sync_wrapper_preserves_multiple_roots_and_native_children(pipeline):
    tracer, _, _, _ = pipeline
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_SYNC)
    token = set_current_golden(Golden(input="one"))
    try:
        with Observer("custom", "__wrapper__"):
            captured = current_trace_context.get()
            for name in ("first", "second"):
                with tracer.start_as_current_span(
                    name, attributes={"gen_ai.operation.name": "invoke_agent"}
                ):
                    with Observer("tool", "native"):
                        pass
        assert [r.name for r in captured.root_spans] == ["first", "second"]
        assert all(r.children[0].name == "native" for r in captured.root_spans)
    finally:
        reset_current_golden(token)


def test_concurrent_goldens_keep_start_ownership(pipeline):
    tracer, processor, _, _ = pipeline
    session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    trace_manager.eval_session = session
    goldens = [Golden(input="a"), Golden(input="b")]

    async def run():
        ready = asyncio.Event()

        async def work(i):
            with tracer.start_as_current_span(
                goldens[i].input,
                attributes={"gen_ai.operation.name": "invoke_agent"},
            ):
                if i == 0:
                    await ready.wait()
                else:
                    ready.set()
                with tracer.start_as_current_span("child"):
                    await asyncio.sleep(0)

        tasks = []
        for i, golden in enumerate(goldens):
            token = set_current_golden(golden)
            try:
                tasks.append(asyncio.create_task(work(i)))
            finally:
                reset_current_golden(token)
        await asyncio.gather(*tasks)

    asyncio.run(run())
    assert len(session.traces_to_evaluate) == 2
    for trace in session.traces_to_evaluate:
        assert (
            trace.root_spans[0].name
            == session.trace_uuid_to_golden[trace.uuid].input
        )
        assert len(trace.root_spans[0].children) == 1
    assert processor._capture.drained()


def test_production_span_does_not_change_route_when_eval_starts(pipeline):
    tracer, processor, _, _ = pipeline
    span = tracer.start_span("production")
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    token = set_current_golden(Golden(input="evaluation"))
    try:
        span.end()
    finally:
        reset_current_golden(token)
    processor._otlp_processor.on_end.assert_called_once()
    assert not trace_manager.eval_session.traces_to_evaluate


def test_background_work_is_not_captured_by_global_eval_flag(pipeline):
    tracer, processor, _, _ = pipeline
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    with tracer.start_as_current_span("unrelated"):
        pass
    processor._otlp_processor.on_end.assert_called_once()
    assert not trace_manager.eval_session.pending_traces


def test_unfinished_span_is_error_and_late_end_cannot_enter_next_run(pipeline):
    tracer, processor, _, _ = pipeline
    session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    trace_manager.eval_session = session
    token = set_current_golden(Golden(input="old"))
    span = tracer.start_span("unfinished")
    reset_current_golden(token)
    assert not processor._capture.drained()
    finish_capture(session)
    assert session.traces_to_evaluate[0].status == TraceSpanStatus.ERRORED
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    span.end()
    assert not trace_manager.eval_session.traces_to_evaluate
    processor._otlp_processor.on_end.assert_not_called()


def test_trace_context_waits_for_ended_spans(pipeline):
    from deepeval.tracing.trace_context import trace

    tracer, _, _, _ = pipeline
    with trace() as captured:
        span = tracer.start_span("delayed")
    trace_manager.post_trace.assert_not_called()
    span.end()
    trace_manager.post_trace.assert_called_once_with(captured)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("[]", []),
        ("broken", [{"role": "user", "content": "legacy"}]),
        ("{}", [{"role": "user", "content": "legacy"}]),
    ],
)
def test_independent_content_fallback(value, expected):
    span = SimpleNamespace(
        attributes={"gen_ai.input.messages": value},
        events=[
            Event("gen_ai.user.message", {"content": "legacy"}),
            Event("gen_ai.choice", {"content": "answer"}),
        ],
    )
    input, output = check_llm_input_from_gen_ai_attributes(span)
    assert input == expected
    assert output == [{"role": "assistant", "content": "answer"}]


def test_confident_fields_decode_and_zero_usage_survives(pipeline):
    tracer, processor, _, _ = pipeline
    with Observer("agent", "parent"):
        captured = current_trace_context.get()
        with tracer.start_as_current_span(
            "chat",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "test",
                "gen_ai.provider.name": "openai",
                "gen_ai.usage.input_tokens": 7,
                "confident.llm.input_token_count": 0,
                "confident.span.input": json.dumps({"query": "x"}),
                "confident.span.output": '""',
            },
        ):
            pass
    child = captured.root_spans[0].children[0]
    assert isinstance(child, LlmSpan)
    assert child.input == {"query": "x"}
    assert child.output == ""
    assert child.input_token_count == 0


def test_observe_inside_bare_otel_entry_promotes_capture(pipeline):
    tracer, processor, _, _ = pipeline
    with tracer.start_as_current_span("otel entry"):
        with Observer("tool", "native child"):
            captured = current_trace_context.get()
    trace_manager.post_trace.assert_called_once_with(captured)
    assert captured.root_spans[0].children[0].name == "native child"
    processor._otlp_processor.on_end.assert_not_called()


def test_parent_ends_before_child_without_losing_tree(pipeline):
    from opentelemetry import trace as otel

    tracer, processor, _, _ = pipeline
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    token = set_current_golden(Golden(input="one"))
    root = tracer.start_span("root")
    child = tracer.start_span("child", context=otel.set_span_in_context(root))
    root.end()
    assert not trace_manager.eval_session.traces_to_evaluate
    child.end()
    reset_current_golden(token)
    captured = trace_manager.eval_session.traces_to_evaluate[0]
    assert captured.root_spans[0].children[0].name == "child"
    assert processor._capture.drained()


def test_two_integrations_share_capture_and_preserve_parentage(pipeline):
    from types import SimpleNamespace
    from deepeval.integrations.agentcore.instrumentator import (
        AgentCoreSpanInterceptor,
    )
    from deepeval.tracing.otel.provider import attach
    from deepeval.tracing.otel.context_aware_processor import (
        ContextAwareSpanProcessor,
    )

    tracer, processor, router, provider = pipeline
    settings = SimpleNamespace(
        name=None,
        thread_id=None,
        user_id=None,
        tags=None,
        metadata=None,
        metric_collection=None,
        test_case_id=None,
        turn_id=None,
        environment="development",
    )
    other_processor = ContextAwareSpanProcessor()
    assert (
        attach(
            provider,
            "agentcore",
            AgentCoreSpanInterceptor(settings),
            other_processor,
        )
        is router
    )
    assert other_processor._capture is processor._capture
    with Observer("agent", "outer"):
        captured = current_trace_context.get()
        with tracer.start_as_current_span(
            "strands", attributes={"gen_ai.operation.name": "invoke_agent"}
        ):
            with provider.get_tracer("agentcore").start_as_current_span(
                "core tool",
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup",
                },
            ):
                pass
    root = captured.root_spans[0]
    assert len(root.children) == 1
    assert len(root.children[0].children) == 1
    assert isinstance(root.children[0].children[0], ToolSpan)
    assert root.children[0].children[0].integration == "AgentCore"
    trace_manager.post_trace.assert_called_once()


def test_capture_cleanup_releases_router_and_live_placeholders(pipeline):
    tracer, processor, router, _ = pipeline
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    token = set_current_golden(Golden(input="one"))
    span = tracer.start_span("unfinished")
    reset_current_golden(token)
    finish_capture(trace_manager.eval_session)
    interceptor = router.adapters["strands"][0]
    assert not interceptor._placeholders
    assert not interceptor._trace_placeholders
    assert not router.active
    span.end()
    processor._otlp_processor.on_end.assert_not_called()


def test_end_in_worker_context_does_not_leak_into_next_trace(pipeline):
    from concurrent.futures import ThreadPoolExecutor

    tracer, processor, _, _ = pipeline
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    token = set_current_golden(Golden(input="first"))
    span = tracer.start_span("first")
    reset_current_golden(token)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(span.end).result()
    token = set_current_golden(Golden(input="second"))
    with tracer.start_as_current_span("second"):
        pass
    reset_current_golden(token)
    traces = trace_manager.eval_session.traces_to_evaluate
    assert [t.root_spans[0].name for t in traces] == ["first", "second"]
    assert processor._capture.drained()


def test_owned_provider_records_eval_but_keeps_production_sampler():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
    from deepeval.tracing.otel.provider import configure_owned_sampling
    from deepeval.tracing.otel.capture import current_eval_owner

    provider = TracerProvider(sampler=ALWAYS_OFF, shutdown_on_exit=False)
    configure_owned_sampling(provider)
    tracer = provider.get_tracer("test")
    assert not tracer.start_span("production").is_recording()
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    token = current_eval_owner.set(trace_manager.eval_session)
    try:
        with tracer.start_as_current_span("evaluation") as span:
            assert span.is_recording()
    finally:
        current_eval_owner.reset(token)
        provider.shutdown()


def test_reconfiguration_keeps_started_span_transport(pipeline):
    tracer, processor, _, _ = pipeline
    before = processor._otlp_processor
    span = tracer.start_span("before reconfiguration")
    processor.reconfigure_api_key("replacement-test-key")
    after = processor._otlp_processor
    span.end()
    with tracer.start_as_current_span("after reconfiguration"):
        pass
    before.on_end.assert_called_once()
    after.on_end.assert_called_once()


def test_evaluation_auth_is_bound_to_entry(pipeline):
    tracer, processor, _, _ = pipeline
    processor._api_key = "entry-test-key"
    trace_manager.eval_session = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    token = set_current_golden(Golden(input="one"))
    with tracer.start_as_current_span("entry"):
        processor._api_key = "later-test-key"
        with tracer.start_as_current_span("child"):
            pass
    reset_current_golden(token)
    assert (
        trace_manager.eval_session.traces_to_evaluate[0].confident_api_key
        == "entry-test-key"
    )


def test_closed_session_context_does_not_create_new_bindings(pipeline):
    from deepeval.tracing.otel.capture import current_eval_owner

    tracer, processor, router, _ = pipeline
    previous = EvalSession(mode=EvalMode.ITERATOR_ASYNC)
    token = current_eval_owner.set(previous)
    try:
        with tracer.start_as_current_span("late task"):
            pass
    finally:
        current_eval_owner.reset(token)
    assert not router.active
    assert not processor._capture.bindings
    processor._otlp_processor.on_end.assert_not_called()
