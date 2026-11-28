from enum import Enum
from typing import Literal


class ExternalSystemAbuseType(Enum):
    DATA_EXFILTRATION = "data_exfiltration"
    COMMUNICATIONS_SPAM = "communications_spam"
    INTERNAL_SPOOFING = "internal_spoofing"


ExternalSystemAbuseTypes = Literal[
    ExternalSystemAbuseType.DATA_EXFILTRATION.value,
    ExternalSystemAbuseType.COMMUNICATIONS_SPAM.value,
    ExternalSystemAbuseType.INTERNAL_SPOOFING.value,
]
