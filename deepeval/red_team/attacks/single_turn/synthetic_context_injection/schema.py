from pydantic import BaseModel


class EnhancedContext(BaseModel):
    domain_logic: str
    input: str


class IsContextValid(BaseModel):
    is_valid_context: bool
