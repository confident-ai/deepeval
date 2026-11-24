from enum import Enum
from typing import Literal


class SystemReconnaissanceType(Enum):
    FILE_METADATA = "file_metadata"
    DATABASE_SCHEMA = "database_schema"
    RETRIEVAL_CONFIG = "retrieval_config"


SystemReconnaissanceTypes = Literal[
    SystemReconnaissanceType.FILE_METADATA.value,
    SystemReconnaissanceType.DATABASE_SCHEMA.value,
    SystemReconnaissanceType.RETRIEVAL_CONFIG.value,
]
