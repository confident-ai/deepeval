from enum import Enum
from typing import Literal


class BiasType(Enum):
    RELIGION = "religion"
    POLITICS = "politics"
    GENDER = "gender"
    RACE = "race"


BiasTypes = Literal[
    BiasType.RELIGION.value,
    BiasType.POLITICS.value,
    BiasType.GENDER.value,
    BiasType.RACE.value,
]
