import pytest

from deepeval.tracing.trace_context import (
    AgentSpanContext,
    LlmSpanContext,
    current_agent_context,
    current_llm_context,
    trace,
)


def test_nested_trace_restores_outer_span_contexts():
    initial_llm_context = current_llm_context.get()
    initial_agent_context = current_agent_context.get()
    outer_llm_context = LlmSpanContext(metric_collection="outer-llm")
    outer_agent_context = AgentSpanContext(metric_collection="outer-agent")
    inner_llm_context = LlmSpanContext(metric_collection="inner-llm")
    inner_agent_context = AgentSpanContext(metric_collection="inner-agent")

    with trace(
        llm_span_context=outer_llm_context,
        agent_span_context=outer_agent_context,
    ):
        with pytest.raises(RuntimeError):
            with trace(
                llm_span_context=inner_llm_context,
                agent_span_context=inner_agent_context,
            ):
                assert current_llm_context.get() is inner_llm_context
                assert current_agent_context.get() is inner_agent_context
                raise RuntimeError

        assert current_llm_context.get() is outer_llm_context
        assert current_agent_context.get() is outer_agent_context

    assert current_llm_context.get() is initial_llm_context
    assert current_agent_context.get() is initial_agent_context


def test_nested_trace_without_override_preserves_outer_span_contexts():
    outer_llm_context = LlmSpanContext(metric_collection="outer-llm")
    outer_agent_context = AgentSpanContext(metric_collection="outer-agent")

    with trace(
        llm_span_context=outer_llm_context,
        agent_span_context=outer_agent_context,
    ):
        with trace():
            assert current_llm_context.get() is outer_llm_context
            assert current_agent_context.get() is outer_agent_context

        assert current_llm_context.get() is outer_llm_context
        assert current_agent_context.get() is outer_agent_context
