from enum import Enum
from typing import Literal


class AgentIdentityAbuseType(Enum):
    AGENT_IMPERSONATION = "agent_impersonation"
    IDENTITY_INHERITANCE = "identity_inheritance"
    CROSS_AGENT_TRUST_ABUSE = "cross_agent_trust_abuse"


AgentIdentityAbuseTypes = Literal[
    AgentIdentityAbuseType.AGENT_IMPERSONATION.value,
    AgentIdentityAbuseType.IDENTITY_INHERITANCE.value,
    AgentIdentityAbuseType.CROSS_AGENT_TRUST_ABUSE.value,
]
