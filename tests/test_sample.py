from deepeval.test_utils import assert_llm_output


def generate_llm_output(input: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output


def test_llm_output():
    input = "What is the customer success phone line?"
    expected_output = "Our customer success phone line is 1200-231-231."
    output = generate_llm_output(input)
    # Now add a simple assert statement for LLM outputs to ensure they match
    # with expected outputs. Etailment is useful for
    assert_llm_output(output, expected_output, metric="entailment")
