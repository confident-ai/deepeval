import warnings
from typing import TYPE_CHECKING, Any

from deepeval.cli.utils import WWW, with_utm

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
        docs_url = with_utm(
            f"{WWW}/docs/integrations/third-party/pydantic-ai",
            medium="python_sdk",
            content="pydantic_ai_agent_deprecation",
        )
        warnings.warn(
            "instrument_pydantic_ai is deprecated and will be removed in a future version. "
            f"Please use the new ConfidentInstrumentationSettings instead. Docs: {docs_url}",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(*args, **kwargs)
