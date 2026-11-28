from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    deception_strategy: str
    input: str


class IsGrayBox(BaseModel):
    is_gray_box: bool
