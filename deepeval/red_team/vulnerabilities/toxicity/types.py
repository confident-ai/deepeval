from enum import Enum
from typing import Literal


class ToxicityType(Enum):
    PROFANITY = "profanity"
    INSULTS = "insults"
    THREATS = "threats"
    MOCKERY = "mockery"


ToxicityTypes = Literal[
    ToxicityType.PROFANITY.value,
    ToxicityType.INSULTS.value,
    ToxicityType.THREATS.value,
    ToxicityType.MOCKERY.value,
]
