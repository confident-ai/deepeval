from pydantic import BaseModel


class EmotionallyEnhancedAttack(BaseModel):
    emotion_strategy: str
    input: str


class IsEmotionalAttackValid(BaseModel):
    is_valid_context: bool
