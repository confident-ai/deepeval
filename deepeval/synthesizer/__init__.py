"""Lazy package init.

Avoids pulling in ``Synthesizer`` (and its ChromaDB chain) just because
something imported ``deepeval.synthesizer.config`` etc.
"""

from typing import Any

__all__ = ["Synthesizer", "Evolution", "PromptEvolution"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .synthesizer import Synthesizer, Evolution, PromptEvolution

        globals().update(
            {
                "Synthesizer": Synthesizer,
                "Evolution": Evolution,
                "PromptEvolution": PromptEvolution,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
