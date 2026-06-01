from importlib import import_module

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.simulator import ConversationSimulator

from metrics import MULTI_TURN_METRICS

MAX_TURNS = 10
ai_app = import_module("ai_app")


simulator = ConversationSimulator(model_callback=ai_app.chatbot_callback)
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")


@pytest.mark.parametrize(
    "test_case",
    simulator.simulate(
        conversational_goldens=dataset.goldens,
        max_user_simulations=MAX_TURNS,
    ),
)
def test_multi_turn(test_case):
    assert_test(test_case=test_case, metrics=MULTI_TURN_METRICS)
