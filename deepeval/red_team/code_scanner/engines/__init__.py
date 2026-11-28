import importlib.util
import os
from typing import Optional

from .base import (
    ENV_KEY,
    HARNESS_PROVIDERS,
    IMPORT_NAME,
    PROVIDERS,
    ScanEngine,
)
from .claude import ClaudeAgentEngine
from .codex import CodexEngine
from .cursor import CursorEngine


def resolve_provider(provider: Optional[str]) -> str:
    """
    Resolve the scan provider.

    An explicit value is validated and returned as-is. Otherwise we infer from
    the API key that is set, but only pick a harness whose SDK is actually
    installed; if none qualifies we fall back to the built-in `deepeval` engine
    (which also uses OPENAI_API_KEY), so a bare install keeps working.
    """
    if provider:
        p = provider.strip().lower()
        if p not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose one of: "
                f"{', '.join(PROVIDERS)}."
            )
        return p

    for harness in HARNESS_PROVIDERS:
        if os.getenv(ENV_KEY[harness]) and (
            importlib.util.find_spec(IMPORT_NAME[harness]) is not None
        ):
            return harness
    return "deepeval"


def build_engine(
    provider: str, model: Optional[str] = None
) -> Optional[ScanEngine]:
    """Build the engine for a provider, or None for the built-in deepeval judge."""
    if provider == "deepeval":
        return None
    if provider == "codex":
        return CodexEngine(model)
    if provider == "claude-code":
        return ClaudeAgentEngine(model)
    if provider == "cursor":
        return CursorEngine(model)
    raise ValueError(f"Unknown provider '{provider}'.")


__all__ = [
    "PROVIDERS",
    "ScanEngine",
    "CodexEngine",
    "ClaudeAgentEngine",
    "CursorEngine",
    "build_engine",
    "resolve_provider",
]
