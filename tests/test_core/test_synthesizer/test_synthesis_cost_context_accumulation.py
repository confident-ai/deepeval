"""Regression tests for synthesis_cost accumulation of context construction.

The docs-based generation paths used `if self.synthesis_cost:` before adding
`context_generator.total_cost`. `synthesis_cost` starts at 0 for native models,
so the condition is falsy exactly when it needs to do the first accumulation —
the context-construction cost was never counted (every other accumulation site
in the file already uses `is not None`).
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import deepeval.synthesizer.synthesizer as synth_mod
from deepeval.synthesizer.synthesizer import Synthesizer
from tests.test_core.stubs import DummyModel, DummyEvolutionConfig

FAKE_CONTEXT_COST = 12.34


@contextmanager
def _null_progress_context(**kwargs):
    yield (_NullProgress(), None)


class _NullProgress:
    """Minimal stand-in for rich Progress: usable as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def remove_task(self, task_id):
        pass


class FakeContextGenerator:
    total_cost = FAKE_CONTEXT_COST
    total_chunks = 1
    pbar_generate_contexts_id = None
    pbar_chunk_docs_id = None
    pbar_load_docs_id = None

    def __init__(self, **kwargs):
        pass

    def generate_contexts(self, **kwargs):
        return ([["ctx chunk one"]], ["doc.pdf"], [0.9])

    async def a_generate_contexts(self, **kwargs):
        return ([["ctx chunk one"]], ["doc.pdf"], [0.9])


def _make_synthesizer(using_native_model: bool) -> Synthesizer:
    s = Synthesizer.__new__(Synthesizer)
    s.model = DummyModel()
    s.evolution_config = DummyEvolutionConfig()
    s.synthetic_goldens = []
    s.synthesis_cost = None
    s.cost_tracking = False
    s.using_native_model = using_native_model
    s.async_mode = False
    return s


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        critic_model=None,
        encoding='utf-8',
        embedder=SimpleNamespace(get_model_name=lambda: 'fake-embedder'),
        chunk_size=100,
        chunk_overlap=10,
        context_quality_threshold=0.5,
        context_similarity_threshold=0.5,
        max_retries=3,
        max_contexts_per_document=1,
        min_contexts_per_document=1,
        max_context_length=1000,
        min_context_length=10,
        allow_cross_file_contexts=False,
        target_files_per_context=None,
        max_files_per_context=None,
    )


def test_sync_docs_cost_counts_context_construction(monkeypatch):
    monkeypatch.setattr(synth_mod, 'synthesizer_progress_context', _null_progress_context)
    monkeypatch.setattr(synth_mod, 'ContextGenerator', FakeContextGenerator)

    s = _make_synthesizer(using_native_model=True)
    monkeypatch.setattr(
        s, 'generate_goldens_from_contexts', lambda **kwargs: []
    )

    s.generate_goldens_from_docs(
        document_paths=['doc.pdf'],
        max_goldens_per_context=1,
        context_construction_config=_make_config(),
    )

    assert s.synthesis_cost == pytest.approx(FAKE_CONTEXT_COST), (
        'context construction cost must be added even when synthesis_cost starts at 0'
    )


async def test_async_docs_cost_counts_context_construction(monkeypatch):
    monkeypatch.setattr(synth_mod, 'synthesizer_progress_context', _null_progress_context)
    monkeypatch.setattr(synth_mod, 'ContextGenerator', FakeContextGenerator)

    s = _make_synthesizer(using_native_model=True)

    async def fake_goldens_from_contexts(**kwargs):
        return []

    monkeypatch.setattr(s, 'a_generate_goldens_from_contexts', fake_goldens_from_contexts)

    await s.a_generate_goldens_from_docs(
        document_paths=['doc.pdf'],
        max_goldens_per_context=1,
        context_construction_config=_make_config(),
    )

    assert s.synthesis_cost == pytest.approx(FAKE_CONTEXT_COST), (
        'context construction cost must be added even when synthesis_cost starts at 0'
    )


async def test_non_native_model_keeps_cost_disabled(monkeypatch):
    """Cost tracking stays off (None) for non-native models after the fix."""
    monkeypatch.setattr(synth_mod, 'synthesizer_progress_context', _null_progress_context)
    monkeypatch.setattr(synth_mod, 'ContextGenerator', FakeContextGenerator)

    s = _make_synthesizer(using_native_model=False)

    async def fake_goldens_from_contexts(**kwargs):
        return []

    monkeypatch.setattr(s, 'a_generate_goldens_from_contexts', fake_goldens_from_contexts)

    await s.a_generate_goldens_from_docs(
        document_paths=['doc.pdf'],
        max_goldens_per_context=1,
        context_construction_config=_make_config(),
    )

    assert s.synthesis_cost is None
