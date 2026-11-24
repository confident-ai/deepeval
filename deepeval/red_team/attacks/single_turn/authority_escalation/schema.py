from pydantic import BaseModel


class EnhancedAuthorityAttack(BaseModel):
    authority_role: str
    input: str


class IsAuthorityValid(BaseModel):
    is_valid_authority: bool
