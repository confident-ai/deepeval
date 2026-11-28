from enum import Enum
from typing import Literal


class RBACType(Enum):
    ROLE_BYPASS = "role_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNAUTHORIZED_ROLE_ASSUMPTION = "unauthorized_role_assumption"


RBACTypes = Literal[
    RBACType.ROLE_BYPASS.value,
    RBACType.PRIVILEGE_ESCALATION.value,
    RBACType.UNAUTHORIZED_ROLE_ASSUMPTION.value,
]
