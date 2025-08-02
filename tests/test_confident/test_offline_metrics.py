from deepeval.tracing.offline_evals import (
    evaluate_span,
    evaluate_trace,
    evaluate_thread,
)
import os

span_id = os.getenv("SPAN_ID")
trace_id = os.getenv("TRACE_ID")
thread_id = os.getenv("THREAD_ID")


def test_evaluate_span_with_custom_metrics():
    evaluate_span(span_id, "My Metrics")
    assert True


def test_evaluate_trace_with_custom_metrics():
    evaluate_trace(trace_id, "My Metrics")
    assert True


def test_evaluate_thread_with_custom_metrics():
    evaluate_thread(thread_id, "My Metrics")
    assert True


def test_evaluate_span_with_task_completion():
    evaluate_span(span_id, "Task Completion")
    assert True


def test_evaluate_trace_with_task_completion():
    evaluate_trace(trace_id, "Task Completion")
    assert True
