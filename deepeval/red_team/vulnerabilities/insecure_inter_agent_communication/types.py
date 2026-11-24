from enum import Enum
from typing import Literal


class InsecureInterAgentCommunicationType(Enum):
    MESSAGE_SPOOFING = "message_spoofing"
    MESSAGE_INJECTION = "message_injection"
    AGENT_IN_THE_MIDDLE = "agent_in_the_middle"


InsecureInterAgentCommunicationTypes = Literal[
    InsecureInterAgentCommunicationType.MESSAGE_SPOOFING.value,
    InsecureInterAgentCommunicationType.MESSAGE_INJECTION.value,
    InsecureInterAgentCommunicationType.AGENT_IN_THE_MIDDLE.value,
]
