from enum import Enum
from typing import List, Optional, Union

from deepteam.test_case.test_case import RTTurn


class StopReason(str, Enum):
    SHIFT_DETECTED = "SHIFT_DETECTED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    SIMULATOR_REFUSED = "SIMULATOR_REFUSED"
    SIMULATION_ERROR = "SIMULATION_ERROR"
    RUNTIME_EXCEEDED = "RUNTIME_EXCEEDED"


# Reasons that mean the run was cut short by our side, not by the target holding
# out — a conversation that ends this way says nothing about the target.
_INCOMPLETE_STOP_REASONS = frozenset(
    {
        StopReason.SIMULATOR_REFUSED,
        StopReason.SIMULATION_ERROR,
        StopReason.RUNTIME_EXCEEDED,
    }
)


def is_progression_completed(reason: Optional[Union[StopReason, str]]) -> bool:
    if reason is None:
        return True
    value = reason.value if isinstance(reason, StopReason) else reason
    return value not in {r.value for r in _INCOMPLETE_STOP_REASONS}


def default_stop_detail(
    reason: StopReason, turns_spent: Optional[int] = None
) -> str:
    if reason is StopReason.SHIFT_DETECTED:
        return (
            "The target's behavior shifted on this turn, ending the "
            "conversation early."
        )
    if reason is StopReason.BUDGET_EXHAUSTED:
        if turns_spent:
            return (
                f"The target held for its full budget of {turns_spent} "
                "turn(s). No behavioral shift was detected at any turn."
            )
        return (
            "The target held for its full turn budget. No behavioral shift "
            "was detected at any turn."
        )
    if reason is StopReason.SIMULATOR_REFUSED:
        return (
            "The simulator model refused to write the next attack, so the "
            "conversation ended before the turn budget was spent. This says "
            "nothing about how long the target resisted."
        )
    if reason is StopReason.SIMULATION_ERROR:
        return (
            "The attack progression failed before the turn budget was spent, "
            "so this conversation's length does not reflect the target's "
            "resistance."
        )
    return (
        "The attack hit its runtime backstop before exhausting its turn "
        "budget, so this conversation's length does not reflect the "
        "target's resistance."
    )


def mark_stop(
    turns: List[RTTurn],
    reason: Union[StopReason, str],
    detail: Optional[str] = None,
    turns_spent: Optional[int] = None,
) -> List[RTTurn]:
    if not turns:
        return turns

    if isinstance(reason, StopReason):
        category = reason
    else:
        try:
            category = StopReason(reason)
        except ValueError:
            category = None

    turns[-1].stopping_category = (
        category.value if category is not None else str(reason)
    )
    if detail:
        turns[-1].stopping_reason = detail
    elif category is not None:
        turns[-1].stopping_reason = default_stop_detail(category, turns_spent)
    else:
        turns[-1].stopping_reason = str(reason)
    return turns


def get_stopping_category(turns: Optional[List[RTTurn]]) -> Optional[str]:
    if not turns:
        return None
    return turns[-1].stopping_category


def get_stopping_reason(turns: Optional[List[RTTurn]]) -> Optional[str]:
    if not turns:
        return None
    return turns[-1].stopping_reason
