from typing import Union, List

from deepeval.test_case import LLMTestCase, ConversationalTestCase


def check_valid_test_cases_type(
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
):
    llm_test_case_count = 0
    conversational_test_case_count = 0
    for test_case in test_cases:
        if isinstance(test_case, LLMTestCase):
            llm_test_case_count += 1
        else:
            conversational_test_case_count += 1

    if llm_test_case_count > 0 and conversational_test_case_count > 0:
        raise ValueError(
            "You cannot supply a mixture of `LLMTestCase`(s) and `ConversationalTestCase`(s) as the list of test cases."
        )
