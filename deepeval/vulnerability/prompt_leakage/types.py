from enum import Enum


class PromptLeakageType(Enum):
    SECRETS_AND_CREDENTIALS = "Secrets and Credentials"
    INSTRUCTIONS = "Instructions"
    GUARDS = "Guards"
    PERMISSIONS_AND_ROLES = "Permissions and Roles Leakage"
