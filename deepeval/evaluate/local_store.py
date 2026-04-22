"""Local file system storage for deepeval test runs.

Persists each `evaluate()` / `evals_iterator()` call as a
`test_run_<YYYYMMDD_HHMMSS>.json` file inside a user-chosen folder. AI tools
(Cursor, Claude Code) can `ls` the folder and read the raw test runs directly:
`TestRun.hyperparameters`, `TestRun.prompts`, per-test-case scores, and metric
reasons all live inside each file via the existing pydantic serialization.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Optional

from deepeval.test_run.test_run import TestRun, TestRunEncoder
from deepeval.utils import is_read_only_env

portalocker = None
if not is_read_only_env():
    try:
        import portalocker
    except Exception as e:  # pragma: no cover - environment dependent
        print(
            f"Warning: failed to import portalocker in local_store: {e}",
            file=sys.stderr,
        )


_LOCK_FILENAME = ".test_run.lock"


def resolve_target_dir(
    results_folder: Optional[str],
    results_subfolder: Optional[str] = None,
) -> Optional[Path]:
    """Resolve where `test_run_*.json` files should be written.

    - `results_folder` set → `Path(results_folder) / results_subfolder` (when
      subfolder is non-empty) else `Path(results_folder)`.
    - `results_folder` unset but `DEEPEVAL_RESULTS_FOLDER` env var set → use
      the env var (backwards compat with existing `save_test_run_locally`).
    - Neither set → `None` (local-store is a no-op).
    """
    folder = results_folder or os.getenv("DEEPEVAL_RESULTS_FOLDER")
    if not folder:
        return None

    base = Path(folder)
    if results_subfolder:
        return base / results_subfolder
    return base


def resolve_test_run_path(target_dir: Path) -> Path:
    """Resolve the exact `test_run_*.json` path inside `target_dir`.

    Base name: `test_run_<YYYYMMDD_HHMMSS>.json` — matches the existing
    `DEEPEVAL_RESULTS_FOLDER` timestamp format byte-for-byte, just with the
    `.json` extension the original code forgot.

    If that exact path already exists (same-second collision), appends
    `_2`, `_3`, … until unique. Callers should hold the lock returned by
    `_acquire_lock(target_dir)` when racing writers are possible.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = target_dir / f"test_run_{ts}.json"
    if not candidate.exists():
        return candidate

    n = 2
    while True:
        candidate = target_dir / f"test_run_{ts}_{n}.json"
        if not candidate.exists():
            return candidate
        n += 1


def write_test_run(
    target_dir: Path,
    test_run: TestRun,
) -> Path:
    """Write `test_run` to `target_dir` as `test_run_<YYYYMMDD_HHMMSS>.json`.

    Uses `TestRunEncoder` (and `model_dump(by_alias=True, exclude_none=True)`)
    so the serialized form matches the `.deepeval/.temp_test_run_data.json`
    format byte-for-byte — the same payload Confident AI uploads.

    Returns the path written. Raises on filesystem errors; callers should
    wrap this in `try/except` so local-save failures never break the eval.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    if portalocker is not None:
        lock_path = target_dir / _LOCK_FILENAME
        with portalocker.Lock(str(lock_path), mode="w"):
            path = resolve_test_run_path(target_dir)
            _dump(test_run, path)
    else:  # pragma: no cover - portalocker is pinned in requirements
        path = resolve_test_run_path(target_dir)
        _dump(test_run, path)

    return path


def _dump(test_run: TestRun, path: Path) -> None:
    try:
        body = test_run.model_dump(by_alias=True, exclude_none=True)
    except AttributeError:
        body = test_run.dict(by_alias=True, exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, cls=TestRunEncoder)
        f.flush()
        os.fsync(f.fileno())
