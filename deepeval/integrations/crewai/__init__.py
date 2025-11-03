from typing import TYPE_CHECKING

__all__ = ["instrument_crewai", "Crew", "Agent", "LLM", "tool"]

if TYPE_CHECKING:
    # This allows the type checkers to see real symbols even while keeping the runtime lazy
    from .handler import instrument_crewai as instrument_crewai
    from .subs import (
        DeepEvalCrew as Crew,
        DeepEvalAgent as Agent,
        DeepEvalLLM as LLM,
    )
    from .tool import tool as tool


def __getattr__(name: str):
    if name == "instrument_crewai":
        from .handler import instrument_crewai

        return instrument_crewai
    if name in ("Crew", "Agent", "LLM"):
        from .subs import (
            DeepEvalCrew as _Crew,
            DeepEvalAgent as _Agent,
            DeepEvalLLM as _LLM,
        )

        return {"Crew": _Crew, "Agent": _Agent, "LLM": _LLM}[name]
    if name == "tool":
        from .tool import tool

        return tool
    raise AttributeError(name)
