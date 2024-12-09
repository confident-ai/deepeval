from enum import Enum


class ToxicityType(Enum):
    PROFANITY = "Profanity"
    INSULTS = "Insults"
    THREATS = "Threats"
    MOCKERY = "Mockery"
