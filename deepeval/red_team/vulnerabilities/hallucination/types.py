from enum import Enum
from typing import Literal


class HallucinationType(Enum):
    FAKE_CITATIONS = "fake_citations"
    FAKE_APIS = "fake_apis"
    FAKE_ENTITIES = "fake_entities"
    FAKE_STATISTICS = "fake_statistics"


HallucinationTypes = Literal[
    HallucinationType.FAKE_CITATIONS.value,
    HallucinationType.FAKE_APIS.value,
    HallucinationType.FAKE_ENTITIES.value,
    HallucinationType.FAKE_STATISTICS.value,
]
