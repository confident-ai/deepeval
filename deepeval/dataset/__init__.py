from importlib import import_module
from typing import TYPE_CHECKING

from deepeval.contextvars import get_current_golden
from .expectations import Expectations

if TYPE_CHECKING:
    from .dataset import EvaluationDataset
    from .golden import (
        Golden,
        ConversationalGolden,
        Persona,
        BackgroundNoiseSettings,
        InterruptionBehavior,
    )

__all__ = [
    "Expectations",
    "EvaluationDataset",
    "Golden",
    "ConversationalGolden",
    "Persona",
    "BackgroundNoiseSettings",
    "InterruptionBehavior",
    "get_current_golden",
]


def __getattr__(name: str):
    # Test cases import dataset.expectations while they are initializing.
    # Loading goldens or EvaluationDataset here eagerly would import those
    # partially initialized test cases again.
    if name == "EvaluationDataset":
        module = import_module(".dataset", __name__)
    elif name in {
        "Golden",
        "ConversationalGolden",
        "Persona",
        "BackgroundNoiseSettings",
        "InterruptionBehavior",
    }:
        module = import_module(".golden", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
