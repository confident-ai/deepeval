from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden

from metrics import SINGLE_TURN_TRACE_METRICS


ai_app = import_module("ai_app")


dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")


@pytest.mark.parametrize("golden", dataset.goldens)
def test_single_turn_tracing(golden: Golden):
    ai_app.run_traced_ai_app(golden.input)
    assert_test(golden=golden, metrics=SINGLE_TURN_TRACE_METRICS)
