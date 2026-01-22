# ruff: noqa: E402
# SQLite version shim - must be at the very top before any crewai/chromadb imports
# ChromaDB requires sqlite >= 3.35.0, but some environments ship older versions.
# If pysqlite3 is available and system sqlite is too old, we swap it in.
from __future__ import annotations

import importlib
import sys
from typing import Tuple

MIN_SQLITE = (3, 35, 0)


def _ensure_sqlite() -> Tuple[bool, Tuple[int, int, int], bool]:
    """
    Ensure sqlite meets minimum version, using pysqlite3 if needed.

    Returns:
        (ok, version_info)
        - ok: True if sqlite version is adequate
        - version_info: the active sqlite version tuple
    """
    import sqlite3

    version_info = sqlite3.sqlite_version_info

    if version_info >= MIN_SQLITE:
        return (True, version_info, False)

    # System sqlite is too old - try pysqlite3
    try:
        import pysqlite3

        sys.modules["sqlite3"] = pysqlite3
        # Re-import to get the fresh binding
        sqlite3 = importlib.import_module("sqlite3")
        version_info = sqlite3.sqlite_version_info

        if version_info >= MIN_SQLITE:
            return (True, version_info, True)
        else:
            # pysqlite3 is also too old (unlikely but possible)
            return (False, version_info, True)
    except ImportError:
        # pysqlite3 not installed
        return (False, version_info, False)


# Run sqlite check at module load (before any crewai/chromadb imports)
_sqlite_ok, _sqlite_version_info, _ = _ensure_sqlite()

# Now safe to import other modules
import os

import pytest

from tests.test_integrations.utils import assert_trace_json

_current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_current_dir, "knowledge_retriever.json")

# Content for knowledge source - just a constant, no CrewAI objects at module level
_KNOWLEDGE_CONTENT = (
    "Users name is John. He is 30 years old and lives in San Francisco."
)


def _check_sqlite_or_skip() -> None:
    """Skip test if sqlite is too old and cannot be fixed."""
    if not _sqlite_ok:
        version_str = ".".join(map(str, _sqlite_version_info))
        min_str = ".".join(map(str, MIN_SQLITE))
        pytest.skip(
            f"SQLite >= {min_str} required by ChromaDB (found {version_str}). "
            f"Install with `pip install pysqlite3-binary` or upgrade system SQLite."
        )


def _handle_chromadb_panic(e: BaseException) -> None:
    """Skip test if chromadb Rust bindings panic, otherwise re-raise."""
    if isinstance(e, (KeyboardInterrupt, SystemExit)):
        raise
    err_name = type(e).__name__
    err_module = type(e).__module__ or ""
    if "Panic" in err_name or "pyo3" in err_module.lower():
        pytest.skip(
            f"ChromaDB Rust bindings incompatible with this environment: {err_name}"
        )
    raise


@assert_trace_json(json_path)
def test_knowledge_retriever():
    # Check sqlite compatibility before importing knowledge-related modules
    _check_sqlite_or_skip()

    # Import CrewAI components here to avoid module-level chromadb initialization
    from crewai import Agent, Crew, LLM, Process, Task
    from crewai.knowledge.source.string_knowledge_source import (
        StringKnowledgeSource,
    )

    # Create a knowledge source - this may fail with chromadb Rust binding issues
    # in certain environments even when sqlite version is adequate
    try:
        string_source = StringKnowledgeSource(content=_KNOWLEDGE_CONTENT)
    except BaseException as e:
        _handle_chromadb_panic(e)

    # Create an LLM with a temperature of 0 to ensure deterministic outputs
    llm = LLM(model="gpt-4o-mini", temperature=0)

    # Create an agent with the knowledge store
    agent = Agent(
        role="About User",
        goal="You know everything about the user.",
        backstory="You are a master at understanding people and their preferences.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    task = Task(
        description="Answer the following questions about the user: {question}",
        expected_output="An answer to the question.",
        agent=agent,
    )

    # Creating crew with knowledge_sources may also trigger chromadb initialization
    try:
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential,
            knowledge_sources=[
                string_source
            ],  # Enable knowledge by adding the sources here
        )
    except BaseException as e:
        _handle_chromadb_panic(e)

    crew.kickoff(
        inputs={"question": "What city does John live in and how old is he?"}
    )
