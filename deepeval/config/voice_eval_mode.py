"""Resolves whether a voice metric's flagged events get a second check by an
audio decision model: `DEEPEVAL_VOICE_EVAL_MODE`.

- `deterministic`: the metric's own detector decides. No model is built or
  called, so a passed `audio_decision_model` is not used. The default.
- `hybrid`: the detector flags events and an audio decision model hears a
  clip of each one and dismisses the ones it judges not real.

Precedence: an explicit `eval_mode=` on the metric, then the setting, then
`deterministic`. Separate from `DEEPEVAL_EVAL_MODE`, which picks who decides
in LLM-as-a-judge metrics.
"""

from enum import Enum
from typing import Literal, Optional

VOICE_EVAL_MODE_ENV_VAR = "DEEPEVAL_VOICE_EVAL_MODE"

VoiceEvalModeName = Literal["deterministic", "hybrid"]


class VoiceEvalMode(str, Enum):
    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"

    def __str__(self) -> str:
        return self.value


SUPPORTED_VOICE_EVAL_MODES = tuple(m.value for m in VoiceEvalMode)

DEFAULT_VOICE_EVAL_MODE = VoiceEvalMode.DETERMINISTIC


def normalize_voice_eval_mode(value: Optional[str]) -> Optional[VoiceEvalMode]:
    """Return the `VoiceEvalMode` for a raw value, or `None` if it is unset,
    blank or unrecognised. Case-insensitive, whitespace ignored."""
    if value is None:
        return None
    if isinstance(value, VoiceEvalMode):
        return value
    try:
        return VoiceEvalMode(str(value).strip().lower())
    except ValueError:
        return None


def resolve_voice_eval_mode(
    override: Optional[VoiceEvalModeName] = None,
) -> VoiceEvalMode:
    """The effective eval mode for a voice metric.

    `override` is the metric's `eval_mode` argument and wins outright. An
    unrecognised override is a `ValueError` (the user typed it), whereas an
    unrecognised setting silently means unset."""
    if override is not None:
        mode = normalize_voice_eval_mode(override)
        if mode is None:
            raise ValueError(
                f"Unsupported eval_mode {override!r}. Valid values: "
                f"{', '.join(SUPPORTED_VOICE_EVAL_MODES)}."
            )
        return mode

    from deepeval.config.settings import get_settings

    configured = normalize_voice_eval_mode(
        get_settings().DEEPEVAL_VOICE_EVAL_MODE
    )
    return configured if configured is not None else DEFAULT_VOICE_EVAL_MODE
