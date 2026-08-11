"""Which model is doing the judging.

Both halves read the source of truth instead of restating it. The provider is
the model's own class name; the model name is checked against the registries in
`deepeval.models.llms.constants` that the library already maintains.

What is *not* taken from the model is `get_model_name()`. The base
implementation returns `self.name`, so a custom `DeepEvalBaseLLM` subclass would
emit an arbitrary user-defined string -- an unbounded-cardinality leak.
"""

from functools import lru_cache
from typing import Any, FrozenSet, Optional, Tuple

from deepeval.telemetry.properties import CUSTOM_PROVIDER, UNKNOWN_MODEL

# Cardinality is bounded by an invariant rather than by a list: a class defined
# under `deepeval.` was written by us and is therefore finite, and everything
# else -- including any user subclass -- collapses to `custom`. A new provider
# needs no change here.
_IN_REPO_PREFIX = "deepeval."


@lru_cache(maxsize=1)
def _known_model_names() -> FrozenSet[str]:
    """Every model name the library ships a registry entry for.

    Pooled across providers rather than checked per provider: the point is only
    to prove a name is one of ours and not a user's private deployment, and
    pooling means a new registry is picked up automatically.
    """
    try:
        from deepeval.models.llms.constants import ModelDataRegistry
        from deepeval.models.llms import constants
    except Exception:
        return frozenset()

    names: set[str] = set()
    for value in vars(constants).values():
        if isinstance(value, ModelDataRegistry):
            names.update(value)
    return frozenset(names)


def _known_model_name(name: Any) -> str:
    if not name or not isinstance(name, str):
        return UNKNOWN_MODEL
    return name if name in _known_model_names() else UNKNOWN_MODEL


def describe_judge(model: Any) -> Tuple[Optional[str], Optional[str]]:
    """Map a judge model instance to a bounded (provider, model) pair."""
    if model is None:
        return None, None
    try:
        model_class = type(model)
        if not (model_class.__module__ or "").startswith(_IN_REPO_PREFIX):
            return CUSTOM_PROVIDER, UNKNOWN_MODEL
        return model_class.__name__, _known_model_name(
            getattr(model, "name", None)
        )
    except Exception:
        return CUSTOM_PROVIDER, UNKNOWN_MODEL
