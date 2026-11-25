from enum import Enum
from typing import Literal


class IllegalActivityType(Enum):
    WEAPONS = "weapons"
    ILLEGAL_DRUGS = "illegal_drugs"
    VIOLENT_CRIME = "violent_crimes"
    NON_VIOLENT_CRIME = "non_violent_crimes"
    SEX_CRIME = "sex_crimes"
    CYBERCRIME = "cybercrime"
    CHILD_EXPLOITATION = "child_exploitation"


IllegalActivityTypes = Literal[
    IllegalActivityType.WEAPONS.value,
    IllegalActivityType.ILLEGAL_DRUGS.value,
    IllegalActivityType.VIOLENT_CRIME.value,
    IllegalActivityType.NON_VIOLENT_CRIME.value,
    IllegalActivityType.SEX_CRIME.value,
    IllegalActivityType.CYBERCRIME.value,
    IllegalActivityType.CHILD_EXPLOITATION.value,
]
