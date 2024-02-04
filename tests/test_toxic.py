"""Tests for toxic
"""


# import pytest
# from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# from deepeval.metrics import ToxicityMetric
# from deepeval import assert_test

# output = "Winners of the FIFA world cup were the French national football team"

# TODO: uncomment once detoxify fixed
# def test_non_toxic():
#     metric = ToxicityMetric(evaluation_params=[LLMTestCaseParams.INPUT])
#     test_case = LLMTestCase(
#         input="Who won the FIFA World Cup in 2018?", actual_output="Don't know."
#     )
#     assert_test(test_case, [metric])


# def test_non_toxic_2():
#     test_case = LLMTestCase(input="Placeholder", actual_output="You're stupid")
#     metric = ToxicityMetric(evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT])
#     with pytest.raises(AssertionError):
#         assert_test(test_case, [metric])


# def test_non_toxic_metric():
#     metric = ToxicityMetric(evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT])
#     test_case = LLMTestCase(input="placeholder", actual_output=output)
#     assert_test(test_case, [metric])
