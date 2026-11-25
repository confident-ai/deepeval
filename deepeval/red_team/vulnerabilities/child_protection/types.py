from enum import Enum
from typing import Literal


class ChildProtectionType(Enum):
    AGE_VERIFICATION = "age_verification"
    DATA_PRIVACY = "data_privacy"
    EXPOSURE_INTERACTION = "exposure_interaction"


ChildProtectionTypes = Literal[
    ChildProtectionType.AGE_VERIFICATION.value,
    ChildProtectionType.DATA_PRIVACY.value,
    ChildProtectionType.EXPOSURE_INTERACTION.value,
]
