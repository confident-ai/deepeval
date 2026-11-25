from enum import Enum


class BFLAType(Enum):
    """
    Enum for BFLA (Broken Function Level Authorization) vulnerability types.

    - Privilege escalation through unauthorized function access
    - Function-level authorization bypass attacks
    - Access control validation failures
    """

    PRIVILEGE_ESCALATION = "privilege_escalation"
    FUNCTION_BYPASS = "function_bypass"
    AUTHORIZATION_BYPASS = "authorization_bypass"


# List of all available types for easy access
BFLA_TYPES = [
    BFLAType.PRIVILEGE_ESCALATION,
    BFLAType.FUNCTION_BYPASS,
    BFLAType.AUTHORIZATION_BYPASS,
]
