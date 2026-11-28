from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    translation_logic: str
    input: str


class IsTranslation(BaseModel):
    """Schema for checking if the prompt is a translation.
    Attributes:
        is_translation (bool): True if the prompt is a translation, False otherwise.
    """

    is_translation: bool
