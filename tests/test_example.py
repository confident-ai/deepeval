"""This is the ideal user flow
"""
from evals.test_utils import assert_match, TestEvalCase, tags


def generate_llm_output(text_input: str):
    return text_input


class TestLLM(TestEvalCase):
    @tags(tags=["customer success"])
    def test_llm_output(self):
        input = "Our customer success phone line is 1200-231-231."
        expected_output = "Our customer success phone line is 1200-231-231."
        output = generate_llm_output(input)
        assert_match(output, expected_output)
