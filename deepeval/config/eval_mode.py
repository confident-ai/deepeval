"""Resolves who decides in an LLM-as-a-judge metric: `DEEPEVAL_EVAL_MODE`.

- `llm`: the evaluation LLM runs the whole chain (extraction, decision,
  reason). The legacy algorithm and the default.
- `hybrid`: the LLM extracts and writes the reason; a System One model (Jev)
  answers the decision points that are wired for it. The LLM is in use
  anyway, so a Jev call that fails at runtime (rate limit, context limit,
  network) hands that one decision to the LLM. A missing key or SDK still
  fails at construction.
- `system_one`: Jev runs the whole metric as one request over the raw test
  case and the reason is deterministic text built from Jev's answers. No LLM
  is built or called, and there is no fallback: the user asked for Jev, so a
  Jev error surfaces as-is and a context overflow says to switch back to
  `llm`. `metric.confidence` reports how decisive the answers were.

Precedence: an explicit `eval_mode=` on the metric, then the setting, then
`llm`. Jev is opted into through the eval mode alone; the `DEEPEVAL_MODE`
feature channel plays no part, so an existing user sees no change and the
eval mode works the same on `stable` and `experimental`. Gate code with
`resolve_eval_mode()` / `EvalMode` members, never by comparing strings.
"""

from enum import Enum
from typing import Literal, Optional

EVAL_MODE_ENV_VAR = "DEEPEVAL_EVAL_MODE"

# The values a metric's `eval_mode` argument, the `DEEPEVAL_EVAL_MODE` setting
# and `deepeval set-eval-mode` accept. One spelling each, so the annotation
# tells the reader exactly what is valid.
EvalModeName = Literal["llm", "hybrid", "system_one"]

# A classifier is already one Choice over its labels, so there is no LLM
# extraction for `hybrid` to keep: its `eval_mode` argument takes these only.
ClassifierEvalModeName = Literal["llm", "system_one"]


class EvalMode(str, Enum):
    LLM = "llm"
    HYBRID = "hybrid"
    SYSTEM_ONE = "system_one"

    def __str__(self) -> str:
        return self.value

    @property
    def uses_system_one(self) -> bool:
        """Whether any Jev call can happen in this mode."""
        return self is not EvalMode.LLM


SUPPORTED_EVAL_MODES = tuple(m.value for m in EvalMode)


def normalize_eval_mode(value: Optional[str]) -> Optional[EvalMode]:
    """Return the `EvalMode` for a raw value, or `None` if it is unset, blank
    or unrecognised. Case-insensitive, whitespace ignored, otherwise exact:
    there are no alternative spellings."""
    if value is None:
        return None
    if isinstance(value, EvalMode):
        return value
    text = str(value).strip().lower()
    try:
        return EvalMode(text)
    except ValueError:
        return None


DEFAULT_EVAL_MODE = EvalMode.LLM


def default_eval_mode() -> EvalMode:
    """The mode used when nothing is configured. Always `llm`: nothing else
    (the feature channel included) turns Jev on implicitly."""
    return DEFAULT_EVAL_MODE


def resolve_eval_mode(override: Optional[EvalModeName] = None) -> EvalMode:
    """The effective eval mode for a metric.

    `override` is the metric's `eval_mode` argument and wins outright. An
    unrecognised override is a `ValueError` (the user typed it), whereas an
    unrecognised setting silently means unset (it must never break loading).
    An `EvalMode` member is accepted at runtime too, for internal callers.
    """
    if override is not None:
        mode = normalize_eval_mode(override)
        if mode is None:
            raise ValueError(
                f"Unsupported eval_mode {override!r}. Valid values: "
                f"{', '.join(SUPPORTED_EVAL_MODES)}."
            )
        return mode

    from deepeval.config.settings import get_settings

    configured = normalize_eval_mode(get_settings().DEEPEVAL_EVAL_MODE)
    return configured if configured is not None else default_eval_mode()


def resolve_classifier_eval_mode(
    override: Optional[ClassifierEvalModeName] = None,
) -> EvalMode:
    """`resolve_eval_mode` for a classifier. Classifiers have no `hybrid`, so
    a `hybrid` setting (from `deepeval set-eval-mode hybrid`) runs them as
    `llm`, silently."""
    mode = resolve_eval_mode(override)
    return EvalMode.LLM if mode is EvalMode.HYBRID else mode
