from dataclasses import dataclass
from typing import Dict, List, Optional
import json

from deepeval.test_case import (
    LLMTestCaseParams,
    ToolCall,
    ArenaTestCase,
    LLMTestCase,
)


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
    input: Optional[str] = None
    expected_output: Optional[str] = None

    def __repr__(self):
        data = {}
        if self.input is not None:
            data["input"] = self.input
        if self.expected_output is not None:
            data["expected_output"] = self.expected_output

        data["arena_test_cases"] = {
            name: repr(contestant)
            for name, contestant in self.contestants.items()
        }
        return json.dumps(data, indent=2)


def format_arena_test_case(
    evaluation_params: List[LLMTestCaseParams], test_case: ArenaTestCase
) -> FormattedArenaTestCase:
    case = next(iter(test_case.contestants.values()))
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
            contestant: construct_formatted_llm_test_case(
                evaluation_params, test_case
            )
            for contestant, test_case in test_case.contestants.items()
        },
    )
    return formatted_test_case


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
