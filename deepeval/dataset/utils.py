from typing import List, Optional, Any
import json
import re

from deepeval.dataset.api import Golden
from deepeval.dataset.golden import ConversationalGolden
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn


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
            tools_called=golden.tools_called,
            expected_tools=golden.expected_tools,
            name=golden.name,
            comments=golden.comments,
            additional_metadata=golden.additional_metadata,
            _dataset_alias=_alias,
            _dataset_id=_id,
            _dataset_rank=index,
        )
        test_cases.append(test_case)
    return test_cases


def convert_convo_test_cases_to_convo_goldens(
    test_cases: List[ConversationalTestCase],
) -> List[ConversationalGolden]:
    goldens = []
    for test_case in test_cases:
        if not test_case.scenario:
            raise ValueError(
                "Please provide a scenario in the test cases to convert them to goldens. Or toggle 'include_test_cases' to False"
            )
        golden = {
            "scenario": test_case.scenario,
            "turns": test_case.turns,
            "expected_outcome": test_case.expected_outcome,
            "user_description": test_case.user_description,
            "context": test_case.context,
        }
        goldens.append(ConversationalGolden(**golden))
    return goldens


def convert_convo_goldens_to_convo_test_cases(
    goldens: List[ConversationalGolden],
    _alias: Optional[str] = None,
    _id: Optional[str] = None,
) -> List[ConversationalTestCase]:
    test_cases = []
    for index, golden in enumerate(goldens):
        test_case = ConversationalTestCase(
            turns=golden.turns or [],
            scenario=golden.scenario,
            user_description=golden.user_description,
            context=golden.context,
            name=golden.name,
            additional_metadata=golden.additional_metadata,
            comments=golden.comments,
            _dataset_alias=_alias,
            _dataset_id=_id,
            _dataset_rank=index,
        )
        test_cases.append(test_case)
    return test_cases


def trimAndLoadJson(input_string: str) -> Any:
    try:
        cleaned_string = re.sub(r",\s*([\]}])", r"\1", input_string.strip())
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {input_string}. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def format_turns(turns):
    formatted_turns = []
    for turn in turns:
        turn_str = f"{turn.role}: {turn.content}"
        if turn.retrieval_context:
            formatted_retrieval_context = "≠".join(turn.retrieval_context)
            turn_str += f" [retrieval_context: {formatted_retrieval_context}]"

        formatted_turns.append(turn_str)

    return "|".join(formatted_turns) if formatted_turns else None


def parse_turns(turns_str):
    turns = []
    if turns_str:
        turn_parts = turns_str.split("|")
        for part in turn_parts:
            role_content = part.split(": ", 1)
            if len(role_content) == 2:
                role, content = role_content
                retrieval_context = None
                if "[" in content:
                    content, context_str = content.split(" [", 1)
                    context_str = context_str.strip("[]").replace(
                        "retrieval_context: ", ""
                    )
                    retrieval_context = context_str.split("≠")
                    retrieval_context = [
                        ctx.strip() for ctx in retrieval_context
                    ]
                turns.append(
                    Turn(
                        role=role,
                        content=content,
                        retrieval_context=retrieval_context,
                    )
                )
            else:
                continue
    return turns
