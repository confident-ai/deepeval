import asyncio

import pytest

import deepteam.code_scanner.code_scanner as cs_module
from deepteam.code_scanner import (
    CodeChunk,
    CodeFinding,
    CodeFindingsList,
    CodeScanner,
)


def _finding(fp="a.py", ln=1):
    return CodeFinding(
        filePath=fp,
        lineStart=ln,
        vulnerability="SSRF",
        vulnerabilityType="port_scanning",
        severity="low",
        reason="x",
    )


@pytest.fixture
def no_model_init(monkeypatch):
    # Skip real model construction so tests need no API key.
    monkeypatch.setattr(cs_module, "initialize_model", lambda m: (m, False))


def _chunks(n, size=40):
    return [
        CodeChunk(file_path=f"f{i}.py", content="y" * size) for i in range(n)
    ]


class TestCodeScanner:
    def test_batching_splits_by_size(self, no_model_init):
        sc = CodeScanner(model=object(), limit=50)
        assert len(sc._make_batches(_chunks(3))) == 3

    def test_batching_packs_small_chunks(self, no_model_init):
        sc = CodeScanner(model=object(), limit=100_000)
        assert len(sc._make_batches(_chunks(5, size=1))) == 1

    def test_scan_dedupes(self, no_model_init, monkeypatch):
        monkeypatch.setattr(
            cs_module,
            "generate",
            lambda *a, **k: CodeFindingsList(findings=[_finding()]),
        )
        sc = CodeScanner(model=object(), limit=50)
        # 3 batches each return the identical finding -> deduped to one.
        assert len(sc.scan(_chunks(3))) == 1

    def test_dedupe_keeps_distinct(self):
        out = CodeScanner._dedupe(
            [_finding(ln=1), _finding(ln=2), _finding(ln=1)]
        )
        assert len(out) == 2

    def test_scan_resilient_to_batch_failure(self, no_model_init, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(cs_module, "generate", boom)
        sc = CodeScanner(model=object(), limit=50)
        # Every batch fails -> no crash, empty result.
        assert sc.scan(_chunks(2)) == []

    def test_a_scan(self, no_model_init, monkeypatch):
        counter = {"n": 0}

        async def fake(*a, **k):
            counter["n"] += 1
            return CodeFindingsList(
                findings=[_finding(fp=f"f{counter['n']}.py")]
            )

        monkeypatch.setattr(cs_module, "a_generate", fake)
        sc = CodeScanner(model=object(), limit=50)
        out = asyncio.run(sc.a_scan(_chunks(3)))
        assert len(out) == 3

    def test_instruction_stored(self, no_model_init):
        sc = CodeScanner(model=object(), instruction="be strict")
        assert sc.instruction == "be strict"
