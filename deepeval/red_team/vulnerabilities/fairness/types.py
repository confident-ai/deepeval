from enum import Enum
from typing import Literal


class FairnessType(Enum):
    EQUALITY_CONSISTENCY = "equality_consistency"
    PROCEDURAL_OPPORTUNITY = "procedural_opportunity"
    TEMPORAL_OUTCOME = "temporal_outcome"


FairnessTypes = Literal[
    FairnessType.EQUALITY_CONSISTENCY.value,
    FairnessType.PROCEDURAL_OPPORTUNITY.value,
    FairnessType.TEMPORAL_OUTCOME.value,
]
