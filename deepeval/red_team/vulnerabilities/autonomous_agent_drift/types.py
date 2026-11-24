from enum import Enum
from typing import Literal


class AutonomousAgentDriftType(Enum):
    GOAL_DRIFT = "goal_drift"
    REWARD_HACKING = "reward_hacking"
    AGENT_COLLUSION = "agent_collusion"
    RUNAWAY_AUTONOMY = "runaway_autonomy"


AutonomousAgentDriftTypes = Literal[
    AutonomousAgentDriftType.GOAL_DRIFT.value,
    AutonomousAgentDriftType.REWARD_HACKING.value,
    AutonomousAgentDriftType.AGENT_COLLUSION.value,
    AutonomousAgentDriftType.RUNAWAY_AUTONOMY.value,
]
