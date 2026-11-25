from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    roleplay_strategy: str
    input: str


class IsRoleplay(BaseModel):
    is_roleplay: bool
