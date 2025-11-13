import warnings
from typing import TYPE_CHECKING
from .subs import is_crewai_installed


__all__ = [
    "instrument_crewai",
    # Export our prefixed names so we don't collide with crewai
    "DeepEvalCrew",
    "DeepEvalAgent",
    "DeepEvalLLM",
    "deepeval_tool",
]

if TYPE_CHECKING:
    # Let type checkers see the real symbols
    from .handler import instrument_crewai as instrument_crewai
    from .tool import deepeval_tool
    from .subs import (
        DeepEvalCrew as DeepEvalCrew,
        DeepEvalAgent as DeepEvalAgent,
        DeepEvalLLM as DeepEvalLLM,
    )

    # legacy names for typing only
    Crew = DeepEvalCrew
    Agent = DeepEvalAgent
    LLM = DeepEvalLLM
    tool = deepeval_tool


def __getattr__(name: str):
    if name == "instrument_crewai":
        from .handler import instrument_crewai

        return instrument_crewai

    if name in ("DeepEvalCrew", "DeepEvalAgent", "DeepEvalLLM"):
        if not is_crewai_installed():
            raise ImportError(
                "crewai is not installed. Install it (e.g., `poetry install --with integrations`) "
                "or add a `crewai` dependency for the current Python version."
            )
        from .subs import DeepEvalCrew, DeepEvalAgent, DeepEvalLLM

        mapping = {
            "DeepEvalCrew": DeepEvalCrew,
            "DeepEvalAgent": DeepEvalAgent,
            "DeepEvalLLM": DeepEvalLLM,
        }
        return mapping[name]

    if name == "deepeval_tool":
        # Always return the decorator function (not the module)
        from .tool import deepeval_tool

        return deepeval_tool

    # Backwards compatible shim support
    if name in ("Crew", "Agent", "LLM"):
        warnings.warn(
            f"deepeval.integrations.crewai.{name} is deprecated; "
            f"use the prefixed DeepEval{name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        alias = {
            "Crew": "DeepEvalCrew",
            "Agent": "DeepEvalAgent",
            "LLM": "DeepEvalLLM",
        }[name]
        return __getattr__(alias)

    if name == "tool":
        warnings.warn(
            "deepeval.integrations.crewai.tool is deprecated; use deepeval_tool instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Hand back the actual decorator function
        return __getattr__("deepeval_tool")

    raise AttributeError(name)
