"""Unit + integration tests for deepeval/evaluate/local_store.py."""

import json
import re
import threading
from pathlib import Path
from typing import Optional

import pytest

from deepeval.evaluate.configs import DisplayConfig
from deepeval.evaluate.local_store import (
    resolve_target_dir,
    resolve_test_run_path,
    write_test_run,
)
from deepeval.test_run.test_run import (
    TestRun as _TestRun,
    TestRunManager as _TestRunManager,
)


FILENAME_RE = re.compile(r"^test_run_\d{8}_\d{6}(?:_(\d+))?\.json$")


def _make_test_run(
    hyperparameters: Optional[dict] = None,
    identifier: Optional[str] = None,
) -> _TestRun:
    return _TestRun(
        identifier=identifier,
        testFile=None,
        testCases=[],
        metricsScores=[],
        hyperparameters=hyperparameters,
        testPassed=None,
        testFailed=None,
    )


class TestResolveTargetDir:
    def test_results_folder_only_flat(self, tmp_path: Path):
        assert resolve_target_dir(str(tmp_path)) == tmp_path

    def test_results_folder_with_subfolder_nests(self, tmp_path: Path):
        got = resolve_target_dir(str(tmp_path), results_subfolder="test_runs")
        assert got == tmp_path / "test_runs"

    def test_empty_subfolder_is_flat(self, tmp_path: Path):
        assert (
            resolve_target_dir(str(tmp_path), results_subfolder="") == tmp_path
        )
        assert (
            resolve_target_dir(str(tmp_path), results_subfolder=None)
            == tmp_path
        )

    def test_env_var_fallback(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_RESULTS_FOLDER", str(tmp_path))
        assert resolve_target_dir(None) == tmp_path

    def test_display_config_takes_precedence_over_env(
        self, tmp_path: Path, monkeypatch
    ):
        other = tmp_path / "from-env"
        monkeypatch.setenv("DEEPEVAL_RESULTS_FOLDER", str(other))
        target = tmp_path / "from-config"
        assert resolve_target_dir(str(target)) == target

    def test_nothing_set_is_none(self, monkeypatch):
        monkeypatch.delenv("DEEPEVAL_RESULTS_FOLDER", raising=False)
        assert resolve_target_dir(None) is None
        assert resolve_target_dir(None, results_subfolder="x") is None


class TestResolveTestRunPath:
    def test_filename_format(self, tmp_path: Path):
        path = resolve_test_run_path(tmp_path)
        assert path.parent == tmp_path
        m = FILENAME_RE.match(path.name)
        assert m is not None, f"unexpected filename {path.name}"
        # No collision suffix on a first call
        assert m.group(1) is None

    def test_same_second_collision_appends_suffix(self, tmp_path: Path):
        first = resolve_test_run_path(tmp_path)
        first.touch()
        second = resolve_test_run_path(tmp_path)
        assert second.name != first.name
        m = FILENAME_RE.match(second.name)
        assert m is not None
        assert m.group(1) == "2"

        second.touch()
        third = resolve_test_run_path(tmp_path)
        assert FILENAME_RE.match(third.name).group(1) == "3"


class TestWriteTestRun:
    def test_round_trips_hyperparameters_and_prompts(self, tmp_path: Path):
        hp = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "top_k": 5,
        }
        test_run = _make_test_run(hyperparameters=hp, identifier="baseline")

        path = write_test_run(tmp_path, test_run)

        assert path.exists()
        assert path.parent == tmp_path
        assert FILENAME_RE.match(path.name) is not None

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["hyperparameters"] == hp
        assert data["identifier"] == "baseline"

    def test_creates_missing_directory(self, tmp_path: Path):
        target = tmp_path / "evals" / "prompt-v3"
        assert not target.exists()
        write_test_run(target, _make_test_run())
        assert target.is_dir()

    def test_never_overwrites_on_same_second_collision(self, tmp_path: Path):
        p1 = write_test_run(tmp_path, _make_test_run(hyperparameters={"t": 0}))
        p2 = write_test_run(tmp_path, _make_test_run(hyperparameters={"t": 1}))
        p3 = write_test_run(tmp_path, _make_test_run(hyperparameters={"t": 2}))

        for p in (p1, p2, p3):
            assert p.exists()
        assert len({p1, p2, p3}) == 3

        # And each file keeps its own hyperparameters (no overwriting)
        payloads = {
            json.loads(p.read_text(encoding="utf-8"))["hyperparameters"]["t"]
            for p in (p1, p2, p3)
        }
        assert payloads == {0, 1, 2}

    def test_concurrent_writes_are_lock_safe(self, tmp_path: Path):
        n = 8
        errors = []

        def worker(i: int):
            try:
                write_test_run(
                    tmp_path,
                    _make_test_run(hyperparameters={"i": i}),
                )
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        files = sorted(
            p for p in tmp_path.iterdir() if p.name.startswith("test_run_")
        )
        assert len(files) == n


class TestDisplayConfigFields:
    def test_new_fields_default_to_none(self):
        cfg = DisplayConfig()
        assert cfg.results_folder is None
        assert cfg.results_subfolder is None

    def test_fields_accept_strings(self):
        cfg = DisplayConfig(
            results_folder="./evals/prompt-v3",
            results_subfolder="test_runs",
        )
        assert cfg.results_folder == "./evals/prompt-v3"
        assert cfg.results_subfolder == "test_runs"


class TestTestRunManagerLocalStoreIntegration:
    """`TestRunManager.save_test_run_locally()` delegates to local_store."""

    def test_writes_via_configure_local_store(self, tmp_path: Path):
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(hyperparameters={"t": 0}))
        mgr.configure_local_store(results_folder=str(tmp_path))

        mgr.save_test_run_locally()

        files = list(tmp_path.glob("test_run_*.json"))
        assert len(files) == 1

    def test_subfolder_nests(self, tmp_path: Path):
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run(hyperparameters={"t": 0}))
        mgr.configure_local_store(
            results_folder=str(tmp_path),
            results_subfolder="test_runs",
        )

        mgr.save_test_run_locally()

        files = list((tmp_path / "test_runs").glob("test_run_*.json"))
        assert len(files) == 1

    def test_env_var_fallback(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DEEPEVAL_RESULTS_FOLDER", str(tmp_path))
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run())

        mgr.save_test_run_locally()

        files = list(tmp_path.glob("test_run_*.json"))
        assert len(files) == 1

    def test_no_config_is_noop(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("DEEPEVAL_RESULTS_FOLDER", raising=False)
        mgr = _TestRunManager()
        mgr.set_test_run(_make_test_run())

        mgr.save_test_run_locally()

        # No test_run_*.json files should be created anywhere under tmp_path
        # (conftest may create a .deepeval sandbox dir, which we ignore).
        assert list(tmp_path.rglob("test_run_*.json")) == []


class TestForLoopFlow:
    """Simulates the developer-facing `for` loop across evaluate() calls.

    We bypass the real eval pipeline (which needs API keys) by driving the
    same post-eval code path directly on the global test run manager —
    configure_local_store + save_test_run_locally — which is what
    evaluate() now does after wrap-up.
    """

    def test_three_iterations_produce_three_files(self, tmp_path: Path):
        target = tmp_path / "evals" / "prompt-v3"

        for temp in [0.0, 0.4, 0.8]:
            mgr = _TestRunManager()
            mgr.set_test_run(
                _make_test_run(
                    hyperparameters={
                        "model": "gpt-4o-mini",
                        "temperature": temp,
                    }
                )
            )
            mgr.configure_local_store(results_folder=str(target))
            mgr.save_test_run_locally()

        files = sorted(target.glob("test_run_*.json"))
        assert len(files) == 3

        temperatures = sorted(
            json.loads(p.read_text(encoding="utf-8"))["hyperparameters"][
                "temperature"
            ]
            for p in files
        )
        assert temperatures == [0.0, 0.4, 0.8]

    def test_chronological_sort_matches_write_order(self, tmp_path: Path):
        """`ls` order (lexicographic) == write order, thanks to the
        timestamp prefix and the `_N` collision suffix."""
        target = tmp_path / "sweep"

        written = []
        for i in range(5):
            mgr = _TestRunManager()
            mgr.set_test_run(_make_test_run(hyperparameters={"i": i}))
            mgr.configure_local_store(results_folder=str(target))
            mgr.save_test_run_locally()
            # read back which file was the latest
            written.append(
                max(
                    target.glob("test_run_*.json"),
                    key=lambda p: p.stat().st_mtime,
                )
            )

        lex_sorted = sorted(target.glob("test_run_*.json"))
        assert lex_sorted == sorted(written, key=lambda p: p.name)
