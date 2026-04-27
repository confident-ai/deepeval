import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe, update_current_span


DATASET_PATH = "tests/evals/.dataset.json"
EVALUATION_MODEL = "EVALUATION_MODEL"

# Attach component-level metrics to the observed span.
SPAN_LEVEL_METRICS = []

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)


def TARGET_APP_ENTRYPOINT(user_input):
    raise NotImplementedError("Replace TARGET_APP_ENTRYPOINT with your component.")


def APP_RESPONSE_ADAPTER(response):
    """Return the component output for span-level evaluation."""
    return response


@observe(metrics=SPAN_LEVEL_METRICS)
def observed_component(user_input: str):
    response = TARGET_APP_ENTRYPOINT(user_input)
    actual_output = APP_RESPONSE_ADAPTER(response)
    update_current_span(
        test_case=LLMTestCase(input=user_input, actual_output=actual_output)
    )
    return actual_output


@pytest.mark.parametrize("golden", dataset.goldens)
def test_single_turn_component(golden):
    observed_component(golden.input)
    assert_test(golden=golden)
