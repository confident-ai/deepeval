from deepeval.evaluate.execute.e2e import (
    _a_execute_conversational_test_cases,
    _a_execute_llm_test_cases,
    a_execute_test_cases,
    execute_test_cases,
)
from deepeval.evaluate.execute.agentic import _a_execute_agentic_test_case
from deepeval.evaluate.execute.loop import (
    _a_evaluate_traces,
    a_execute_agentic_test_cases_from_loop,
    execute_agentic_test_cases_from_loop,
)
from deepeval.evaluate.execute.trace_scope import (
    _assert_test_from_current_trace,
)

# Re-exposed for tests that reach ``exec_mod.trace_manager`` /
# ``exec_mod.global_test_run_manager`` to mutate the shared singletons.
from deepeval.test_run import global_test_run_manager
from deepeval.tracing.tracing import trace_manager

__all__ = [
    # e2e
    "execute_test_cases",
    "a_execute_test_cases",
    "_a_execute_llm_test_cases",
    "_a_execute_conversational_test_cases",
    # agentic
    "_a_execute_agentic_test_case",
    # loop
    "execute_agentic_test_cases_from_loop",
    "a_execute_agentic_test_cases_from_loop",
    "_a_evaluate_traces",
    # trace-scope
    "_assert_test_from_current_trace",
    # shared singletons
    "global_test_run_manager",
    "trace_manager",
]
