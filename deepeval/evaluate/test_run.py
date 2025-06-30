from contextlib import contextmanager
from typing import List

from deepeval.dataset import Golden
from deepeval.tracing import trace_manager

@contextmanager
def evaluate(goldens: List[Golden]):
    run = TestRun(goldens=goldens)
    run.begin()
    try:
        yield from goldens
    finally:
        run.end()
        span_test_cases_map = trace_manager.span_test_cases_map