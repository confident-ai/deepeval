"""Resolves `DEEPEVAL_MODE`: `stable` (default) or `experimental`.

`experimental` opts a user into the newest deepeval features before they are
finalised; anything gated on it may change or break between releases.
`stable` is the default and the fallback for an unset or unrecognised value,
so a typo can never silently enrol someone.

Gate features with `is_experimental()` rather than comparing strings. The list
of current experimental features and where they live is in `EXPERIMENTAL.md`.
"""

import os
import sys
from enum import Enum
from typing import Optional

MODE_ENV_VAR = "DEEPEVAL_MODE"


class DeepEvalMode(str, Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"

    def __str__(self) -> str:
        return self.value


DEFAULT_MODE = DeepEvalMode.STABLE
SUPPORTED_MODES = tuple(m.value for m in DeepEvalMode)

# Shell-friendly spellings accepted by the CLI and env var.
_MODE_ALIASES = {
    "": None,
    "stable": DeepEvalMode.STABLE,
    "experimental": DeepEvalMode.EXPERIMENTAL,
}

_warned_unrecognised: set = set()


def normalize_deepeval_mode(value: Optional[str]) -> Optional[DeepEvalMode]:
    """Return the `DeepEvalMode` for a raw value, or `None` if it is unset,
    blank or unrecognised."""
    if value is None:
        return None
    return _MODE_ALIASES.get(str(value).strip().lower())


def resolve_deepeval_mode() -> DeepEvalMode:
    """`DeepEvalMode.STABLE` (default) or `DeepEvalMode.EXPERIMENTAL`.

    Reads the env var directly (the `.deepeval` keystore is merged into the
    environment when settings load) so the choice can be flipped per-process.
    Unrecognised values fall back to `stable` with a one-time stderr warning.
    """
    raw = os.getenv(MODE_ENV_VAR) or ""
    mode = normalize_deepeval_mode(raw)
    if mode is not None:
        return mode
    if raw.strip() and raw not in _warned_unrecognised:
        _warned_unrecognised.add(raw)
        print(
            f"Warning: unrecognised {MODE_ENV_VAR}={raw!r}; falling back to "
            f"'{DEFAULT_MODE}'. Valid values: {', '.join(SUPPORTED_MODES)}.",
            file=sys.stderr,
        )
    return DEFAULT_MODE


def is_experimental() -> bool:
    """True when the user has opted into experimental deepeval features."""
    return resolve_deepeval_mode() == DeepEvalMode.EXPERIMENTAL
