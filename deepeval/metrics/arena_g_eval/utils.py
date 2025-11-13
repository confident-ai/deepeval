from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import random

from deepeval.test_case import (
    LLMTestCaseParams,
    ToolCall,
    ArenaTestCase,
    LLMTestCase,
)

# List of fake names to use for masking contestant names
FAKE_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
]


@dataclass
class FormattedLLMTestCase:
    actual_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None
    expected_tools: Optional[List[ToolCall]] = None

    def __repr__(self):
        data = {}
        if self.actual_output is not None:
            data["actual_output"] = self.actual_output
        if self.context is not None:
            data["context"] = self.context
        if self.retrieval_context is not None:
            data["retrieval_context"] = self.retrieval_context
        if self.tools_called is not None:
            data["tools_called"] = [repr(tool) for tool in self.tools_called]
        if self.expected_tools is not None:
            data["expected_tools"] = [
                repr(tool) for tool in self.expected_tools
            ]

        return json.dumps(data, indent=2)


@dataclass
class FormattedArenaTestCase:
    contestants: Dict[str, FormattedLLMTestCase]
    dummy_to_real_names: Dict[str, str]
    input: Optional[str] = None
    expected_output: Optional[str] = None

    def __repr__(self):
        data = {}
        if self.input is not None:
            data["input"] = self.input
        if self.expected_output is not None:
            data["expected_output"] = self.expected_output

        # Randomize the order of contestants
        contestant_items = list(self.contestants.items())
        random.shuffle(contestant_items)

        # Use dummy names if mapping is available, otherwise use real names
        if self.dummy_to_real_names:
            # Create reverse mapping from real to dummy names
            real_to_dummy = {
                real: dummy for dummy, real in self.dummy_to_real_names.items()
            }
            data["arena_test_cases"] = {
                real_to_dummy.get(name, name): repr(contestant)
                for name, contestant in contestant_items
            }
        else:
            data["arena_test_cases"] = {
                name: repr(contestant) for name, contestant in contestant_items
            }
        return json.dumps(data, indent=2)


def format_arena_test_case(
    evaluation_params: List[LLMTestCaseParams], test_case: ArenaTestCase
) -> Tuple[FormattedArenaTestCase, Dict[str, str]]:
    case = next(iter([case.test_case for case in test_case.contestants]))

    # Create dummy name mapping
    real_names = list([case.name for case in test_case.contestants])
    available_fake_names = FAKE_NAMES.copy()
    random.shuffle(available_fake_names)

    # Ensure we have enough fake names
    if len(real_names) > len(available_fake_names):
        # If we need more names, create additional ones by adding numbers
        additional_names = [
            f"Contestant{i+1}"
            for i in range(len(real_names) - len(available_fake_names))
        ]
        available_fake_names.extend(additional_names)

    dummy_to_real_names = {}
    for i, real_name in enumerate(real_names):
        dummy_to_real_names[available_fake_names[i]] = real_name

    formatted_test_case = FormattedArenaTestCase(
        input=(
            case.input if LLMTestCaseParams.INPUT in evaluation_params else None
        ),
        expected_output=(
            case.expected_output
            if LLMTestCaseParams.EXPECTED_OUTPUT in evaluation_params
            else None
        ),
        contestants={
            contestant.name: construct_formatted_llm_test_case(
                evaluation_params, contestant.test_case
            )
            for contestant in test_case.contestants
        },
        dummy_to_real_names=dummy_to_real_names,
    )
    return formatted_test_case, dummy_to_real_names


def construct_formatted_llm_test_case(
    evaluation_params: List[LLMTestCaseParams], test_case: LLMTestCase
) -> FormattedLLMTestCase:
    return FormattedLLMTestCase(
        actual_output=(
            test_case.actual_output
            if LLMTestCaseParams.ACTUAL_OUTPUT in evaluation_params
            else None
        ),
        context=(
            test_case.context
            if LLMTestCaseParams.CONTEXT in evaluation_params
            else None
        ),
        retrieval_context=(
            test_case.retrieval_context
            if LLMTestCaseParams.RETRIEVAL_CONTEXT in evaluation_params
            else None
        ),
        tools_called=(
            test_case.tools_called
            if LLMTestCaseParams.TOOLS_CALLED in evaluation_params
            else None
        ),
        expected_tools=(
            test_case.expected_tools
            if LLMTestCaseParams.EXPECTED_TOOLS in evaluation_params
            else None
        ),
    )
