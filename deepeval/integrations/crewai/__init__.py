# ruff: noqa: E402
from __future__ import annotations

import sys

from deepeval.utils import require_dependency


def _is_chroma_sqlite_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return "Chroma" in msg and "sqlite3" in msg and "unsupported version" in msg


try:
    from .handler import instrument_crewai
except RuntimeError as exc:
    if not _is_chroma_sqlite_error(exc):
        raise

    pysqlite3 = require_dependency(
        "pysqlite3",
        provider_label="DeepEval CrewAI integration",
        install_hint="Install it with `pip install pysqlite3-binary`",
    )
    sys.modules["sqlite3"] = pysqlite3

    # Retry once after swapping sqlite3
    from .handler import instrument_crewai


from .subs import (
    DeepEvalCrew as Crew,
    DeepEvalAgent as Agent,
    DeepEvalLLM as LLM,
)
from .tool import tool

__all__ = ["instrument_crewai", "Crew", "Agent", "LLM", "tool"]
