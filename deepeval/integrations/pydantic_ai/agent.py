import warnings
from typing import TYPE_CHECKING, Any

try:
    from pydantic_ai.agent import Agent as _BaseAgent

    is_pydantic_ai_installed = True
except ImportError:
    is_pydantic_ai_installed = False

    class _BaseAgent:
        """Dummy fallback so imports don't crash when pydantic-ai is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # No-op: for compatibility
            pass


if TYPE_CHECKING:
    # For type checkers: use the real Agent if available.
    from pydantic_ai.agent import Agent  # type: ignore[unused-ignore]
else:
    # At runtime we always have some base: real Agent or our dummy.
    # This is just to avoid blow-ups.
    Agent = _BaseAgent


class DeepEvalPydanticAIAgent(Agent):

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "instrument_pydantic_ai is deprecated and will be removed in a future version. "
            "Please use the new ConfidentInstrumentationSettings instead. Docs: https://www.confident-ai.com/docs/integrations/third-party/pydantic-ai",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(*args, **kwargs)
