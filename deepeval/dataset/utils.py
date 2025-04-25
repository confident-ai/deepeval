from typing import List, Optional, Any
from deepeval.dataset.api import Golden, ConversationalGolden
from deepeval.test_case import LLMTestCase, ConversationalTestCase, MLLMTestCase
import json
import re


def convert_test_cases_to_goldens(
    test_cases: List[LLMTestCase],
) -> List[Golden]:
    goldens = []
    for test_case in test_cases:
        golden = {
            "input": test_case.input,
            "actual_output": test_case.actual_output,
            "expected_output": test_case.expected_output,
            "context": test_case.context,
            "retrieval_context": test_case.retrieval_context,
            "tools_called": test_case.tools_called,
            "expected_tools": test_case.expected_tools,
        }
        goldens.append(Golden(**golden))
    return goldens


def convert_goldens_to_test_cases(
    goldens: List[Golden],
    _alias: Optional[str] = None,
    _id: Optional[str] = None,
) -> List[LLMTestCase]:
    test_cases = []
    for index, golden in enumerate(goldens):
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=golden.actual_output,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=golden.retrieval_context,
            additional_metadata=golden.additional_metadata,
            tools_called=golden.tools_called,
            expected_tools=golden.expected_tools,
            name=golden.name,
            comments=golden.comments,
            _dataset_alias=_alias,
            _dataset_id=_id,
            _dataset_rank=index,
        )
        test_cases.append(test_case)
    return test_cases


def convert_convo_goldens_to_convo_test_cases(
    convo_goldens: List[ConversationalGolden],
    _alias: Optional[str] = None,
    _id: Optional[str] = None,
) -> List[ConversationalTestCase]:
    conv_test_cases = []
    for index, convo_golden in enumerate(convo_goldens):
        conv_test_case = ConversationalTestCase(
            additional_metadata=convo_golden.additional_metadata,
            comments=convo_golden.comments,
            turns=convert_goldens_to_test_cases(convo_golden.turns),
            _dataset_alias=_alias,
            _dataset_id=_id,
            _dataset_rank=index,
        )
        conv_test_cases.append(conv_test_case)
    return conv_test_cases


def trimAndLoadJson(input_string: str) -> Any:
    try:
        cleaned_string = re.sub(r",\s*([\]}])", r"\1", input_string.strip())
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {input_string}. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
