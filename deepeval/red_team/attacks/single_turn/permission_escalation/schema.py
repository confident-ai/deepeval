from pydantic import BaseModel


class EnhancedPermission(BaseModel):
    escalation_logic: str
    input: str


class IsPermissionValid(BaseModel):
    is_valid_permission: bool
