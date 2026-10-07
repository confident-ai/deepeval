import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from deepeval.test_case import LLMTestCase


JUDGE_EVAL_VARIABLE_PATTERN = re.compile(
    r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
)
JUDGE_EVAL_ALL_ITEMS = "*"


class JudgeEvalField(Enum):
    INPUT = "input"
    OUTPUT = "output"
    METADATA = "metadata"
    TOOLS_CALLED = "tools_called"


class JudgeEvalRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class JudgeEvalMessage(BaseModel):
    role: JudgeEvalRole
    content: str


class JudgeEvalVariable(BaseModel):
    field: JudgeEvalField
    path: List[Union[int, str]] = []


def extract_variable_names(messages: List[JudgeEvalMessage]) -> List[str]:
    names: List[str] = []
    for message in messages:
        for name in JUDGE_EVAL_VARIABLE_PATTERN.findall(message.content):
            if name not in names:
                names.append(name)
    return names


def validate_messages(
    messages: Optional[List[JudgeEvalMessage]],
) -> List[JudgeEvalMessage]:
    if not messages:
        raise ValueError("JudgeEval needs at least one message.")
    for index, message in enumerate(messages):
        if message.role == JudgeEvalRole.SYSTEM and index != 0:
            raise ValueError(
                "A system message is only allowed as the first message."
            )
    return list(messages)


def validate_variables(
    messages: List[JudgeEvalMessage],
    variables: Optional[Dict[str, JudgeEvalVariable]],
) -> Dict[str, JudgeEvalVariable]:
    variables = dict(variables or {})
    unmapped = [
        name
        for name in extract_variable_names(messages)
        if name not in variables
    ]
    if unmapped:
        raise ValueError(
            "Every variable in the messages needs a mapping. Missing: "
            + ", ".join(unmapped)
        )
    return variables


def validate_score_range(score_range: Tuple[int, int]) -> Tuple[int, int]:
    if len(score_range) != 2:
        raise ValueError("score_range must be a (min, max) pair.")
    minimum, maximum = score_range
    if minimum >= maximum:
        raise ValueError("score_range min must be lower than max.")
    return (minimum, maximum)


def parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def read_field(test_case: LLMTestCase, field: JudgeEvalField) -> Any:
    if field == JudgeEvalField.INPUT:
        return parse_json(test_case.input)
    if field == JudgeEvalField.OUTPUT:
        return parse_json(test_case.actual_output)
    if field == JudgeEvalField.METADATA:
        return test_case.metadata
    if not test_case.tools_called:
        return None
    return [
        tool_call.model_dump(mode="json", exclude_none=True)
        for tool_call in test_case.tools_called
    ]


def resolve_path(value: Any, path: List[Union[int, str]]) -> Any:
    if not path:
        return value

    value = parse_json(value)
    key, rest = path[0], path[1:]

    if key == JUDGE_EVAL_ALL_ITEMS:
        if not isinstance(value, list):
            return None
        matches = [resolve_path(item, rest) for item in value]
        return [match for match in matches if match is not None]

    if isinstance(key, bool):
        return None

    if isinstance(key, int):
        if isinstance(value, list) and -len(value) <= key < len(value):
            return resolve_path(value[key], rest)
        return None

    if isinstance(value, dict) and key in value:
        return resolve_path(value[key], rest)
    return None


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def resolve_variables(
    test_case: LLMTestCase, variables: Dict[str, JudgeEvalVariable]
) -> Dict[str, str]:
    return {
        name: stringify(
            resolve_path(read_field(test_case, variable.field), variable.path)
        )
        for name, variable in variables.items()
    }


def fill_variables(content: str, values: Dict[str, str]) -> str:
    return JUDGE_EVAL_VARIABLE_PATTERN.sub(
        lambda match: values.get(match.group(1), match.group(0)), content
    )


def render_messages(
    messages: List[JudgeEvalMessage], values: Dict[str, str]
) -> str:
    return "\n\n".join(
        f"{message.role.value.capitalize()}:\n{fill_variables(message.content, values)}"
        for message in messages
    )
