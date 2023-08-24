import pytest
from deepeval.test_utils import assert_llm_output
from deepeval.metrics.entailment_metric import EntailmentScoreMetric


def generate_llm_output(query: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output


@pytest.mark.asyncio
async def test_llm_output():
    query = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(query)
    # Now add a simple assert statement for LLM outputs to ensure they match
    # with expected outputs. Etailment is useful for
    assert_llm_output(output, expected_output, metric="entailment")
