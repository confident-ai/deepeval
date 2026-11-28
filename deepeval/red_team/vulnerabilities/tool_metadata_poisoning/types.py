from enum import Enum
from typing import Literal


class ToolMetadataPoisoningType(Enum):
    SCHEMA_MANIPULATION = "schema_manipulation"
    DESCRIPTION_DECEPTION = "description_deception"
    PERMISSION_MISREPRESENTATION = "permission_misrepresentation"
    REGISTRY_POISONING = "registry_poisoning"


ToolMetadataPoisoningTypes = Literal[
    ToolMetadataPoisoningType.SCHEMA_MANIPULATION.value,
    ToolMetadataPoisoningType.DESCRIPTION_DECEPTION.value,
    ToolMetadataPoisoningType.PERMISSION_MISREPRESENTATION.value,
    ToolMetadataPoisoningType.REGISTRY_POISONING.value,
]
