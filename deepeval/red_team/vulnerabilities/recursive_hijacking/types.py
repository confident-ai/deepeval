from enum import Enum


class RecursiveHijackingType(Enum):
    """
    Enum for Recursive Hijacking vulnerability types.

    - Agent fails to preserve its original mission when goals are incrementally redefined
    - Chaining a sequence of intermediate instructions that gradually alter the end goal
    - Introducing cascading sub-goals that conflict with the primary objective
    """

    SELF_MODIFYING_GOALS = "self_modifying_goals"
    RECURSIVE_OBJECTIVE_CHAINING = "recursive_objective_chaining"
    GOAL_PROPAGATION_ATTACKS = "goal_propagation_attacks"


# List of all available types for easy access
RECURSIVE_HIJACKING_TYPES = [
    RecursiveHijackingType.SELF_MODIFYING_GOALS,
    RecursiveHijackingType.RECURSIVE_OBJECTIVE_CHAINING,
    RecursiveHijackingType.GOAL_PROPAGATION_ATTACKS,
]
