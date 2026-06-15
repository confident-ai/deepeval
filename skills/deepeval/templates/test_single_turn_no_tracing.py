from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase

from metrics import SINGLE_TURN_NO_TRACING_METRICS


ai_app = import_module("ai_app")


dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")


@pytest.mark.parametrize("golden", dataset.goldens)
def test_single_turn_no_tracing(golden: Golden):
    actual_output = ai_app.run_ai_app(golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=getattr(golden, "expected_output", None),
        context=getattr(golden, "context", None),
        retrieval_context=getattr(golden, "retrieval_context", None),
    )
    assert_test(
        test_case=test_case,
        metrics=SINGLE_TURN_NO_TRACING_METRICS,
    )
