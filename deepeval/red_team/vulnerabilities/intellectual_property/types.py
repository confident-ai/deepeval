from enum import Enum
from typing import Literal


class IntellectualPropertyType(Enum):
    IMITATION = "imitation"
    COPYRIGHT_VIOLATIONS = "copyright_violations"
    TRADEMARK_INFRINGEMENT = "trademark_infringement"
    PATENT_DISCLOSURE = "patent_disclosure"


IntellectualPropertyTypes = Literal[
    IntellectualPropertyType.IMITATION.value,
    IntellectualPropertyType.COPYRIGHT_VIOLATIONS.value,
    IntellectualPropertyType.TRADEMARK_INFRINGEMENT.value,
    IntellectualPropertyType.PATENT_DISCLOSURE.value,
]
