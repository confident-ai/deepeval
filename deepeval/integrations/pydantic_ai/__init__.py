from .instrumentator import (
    ConfidentInstrumentationSettings,
    DeepEvalInstrumentationSettings,
)
from .otel import instrument_pydantic_ai

__all__ = [
    "DeepEvalInstrumentationSettings",
    # Deprecated alias kept for backward compatibility — emits a
    # ``DeprecationWarning`` on instantiation. Prefer
    # ``DeepEvalInstrumentationSettings`` in new code.
    "ConfidentInstrumentationSettings",
]
