from enum import Enum
from typing import Literal


class PromptLeakageType(Enum):
    SECRETS_AND_CREDENTIALS = "secrets_and_credentials"
    INSTRUCTIONS = "instructions"
    GUARD_EXPOSURE = "guard_exposure"
    PERMISSIONS_AND_ROLES = "permissions_and_roles"


PromptLeakageTypes = Literal[
    PromptLeakageType.SECRETS_AND_CREDENTIALS.value,
    PromptLeakageType.INSTRUCTIONS.value,
    PromptLeakageType.GUARD_EXPOSURE.value,
    PromptLeakageType.PERMISSIONS_AND_ROLES.value,
]
