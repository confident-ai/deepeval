from typing import Dict, List, Optional, Union

from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    Turn,
    MultiTurnParams,
)


def format_turns(
    llm_test_cases: List[LLMTestCase], test_case_params: List[SingleTurnParams]
) -> List[Dict[str, Union[str, List[str]]]]:
    res = []
    for llm_test_case in llm_test_cases:
        dict = {}
        for param in test_case_params:
            value = getattr(llm_test_case, param.value)
            if value:
                dict[param.value] = value
        res.append(dict)
    return res


def convert_turn_to_dict(
    turn: Turn,
    turn_params: Optional[List[MultiTurnParams]] = None,
) -> Dict:
    if turn_params is None:
        turn_params = [MultiTurnParams.CONTENT, MultiTurnParams.ROLE]
    result = {}
    for param in turn_params:
        if param in (
            MultiTurnParams.SCENARIO,
            MultiTurnParams.EXPECTED_OUTCOME,
            MultiTurnParams.METADATA,
            MultiTurnParams.TAGS,
        ):
            continue

        if not hasattr(turn, param.value):
            continue

        value = getattr(turn, param.value)
        if value is not None:
            result[param.value] = value

    return result


def get_turns_in_sliding_window(turns: List[Turn], window_size: int):
    for i in range(len(turns)):
        yield turns[max(0, i - window_size + 1) : i + 1]


def get_unit_interactions(turns: List[Turn]) -> List[List[Turn]]:
    units: List[List[Turn]] = []
    current: List[Turn] = []
    has_user = False

    for turn in turns:
        # Boundary: user after assistant, but only if we've already seen a user in current
        if (
            current
            and current[-1].role == "assistant"
            and turn.role == "user"
            and has_user
        ):
            units.append(current)  # finalize previous unit
            current = [turn]  # start new unit with this user
            has_user = True
            continue

        # Otherwise just accumulate
        current.append(turn)
        if turn.role == "user":
            has_user = True

    # Finalize last unit only if it ends with assistant and includes a user
    if (
        current
        and len(current) > 1
        and current[-1].role == "assistant"
        and has_user
    ):
        units.append(current)

    return units
