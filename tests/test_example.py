"""This is the ideal user flow
"""
from evals.test_utils import assert_exact_match, TestEvalCase, tags


def generate_llm_output(text_input: str):
    return text_input


class TestLLM(TestEvalCase):
    @tags(tags=["customer success"])
    def test_llm_output(self):
        input = "Our customer success phone line is 1200-231-231."
        expected_output = "Hi this is Jack"
        output = generate_llm_output(input)
        assert_exact_match(output, expected_output)

    @tags(tags=["easy example"])
    def test_llm_output_2(self):
        input = "Hi this is Jack"
        expected_output = "Hi this is NOT Jack"
        output = generate_llm_output(input)
        assert_exact_match(output, expected_output)
