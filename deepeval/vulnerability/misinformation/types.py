from enum import Enum


class MisinformationType(Enum):
    FACTUAL_ERRORS = "Factual Errors"
    UNSUPPORTED_CLAIMS = "Unsupported Claims"
    EXPERTISE_MISREPRESENTATION = "Expertise Misrepresentation"
