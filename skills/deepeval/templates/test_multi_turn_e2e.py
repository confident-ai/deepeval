import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Turn


DATASET_PATH = "tests/evals/.dataset.json"
EVALUATION_MODEL = "EVALUATION_MODEL"

# Must use multi-turn conversational metrics, such as conversation completeness,
# role adherence, turn relevancy, goal accuracy, or ConversationalGEval.
END_TO_END_METRICS = []
MAX_TURNS = 10

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)


async def TARGET_APP_ENTRYPOINT(user_input, turns, thread_id):
    raise NotImplementedError(
        "Replace TARGET_APP_ENTRYPOINT with your chatbot."
    )


async def chatbot_callback(input: str, turns=None, thread_id=None):
    response = await TARGET_APP_ENTRYPOINT(input, turns, thread_id)
    content = APP_RESPONSE_ADAPTER(response)
    return Turn(role="assistant", content=content)


def APP_RESPONSE_ADAPTER(response):
    """Return the assistant message content from the chatbot response."""
    return response


simulator = ConversationSimulator(model_callback=chatbot_callback)
test_cases = simulator.simulate(
    conversational_goldens=dataset.goldens,
    max_user_simulations=MAX_TURNS,
)


@pytest.mark.parametrize("test_case", test_cases)
def test_multi_turn(test_case):
    assert_test(test_case=test_case, metrics=END_TO_END_METRICS)
