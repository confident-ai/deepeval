from typing import TYPE_CHECKING, Dict, List

from deepeval.red_team.attacks.multi_turn.progression import Probe
from deepeval.red_team.test_case.test_case import RTTurn

if TYPE_CHECKING:
    from deepeval.red_team.orchestration.schema import CriteriaVerdict


def as_chat_history(turns: List[RTTurn]) -> List[Dict[str, str]]:
    return [{"role": turn.role, "content": turn.content} for turn in turns]


def as_current_turn(probe: Probe) -> Dict[str, str]:
    return {"user": probe.input, "assistant": probe.output.content}


def copy_turns(turns: List[RTTurn]) -> List[RTTurn]:
    # Fresh copies without stop markers, so a conversation can seed another
    # without the seed's markers showing up mid-history, and without one
    # progression's finalize() marking turns another progression shares.
    return [
        turn.model_copy(
            update={"stopping_reason": None, "stopping_category": None}
        )
        for turn in turns
    ]


def weighted_score(verdicts: List["CriteriaVerdict"]) -> float:
    # A 0-1 average where turn n weighs n, so recent turns count the most.
    if not verdicts:
        return 0.0
    weights = range(1, len(verdicts) + 1)
    return sum(
        weight * verdict.unit_score
        for weight, verdict in zip(weights, verdicts)
    ) / sum(weights)
