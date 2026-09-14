"""`deepeval inspect [PATH]` Typer command.

Heavy imports (Textual, pyperclip) are deferred until invocation so
`deepeval.cli.main` stays cheap and users without the optional extra
get a clean install hint instead of a cryptic ImportError.

Sources can be a `test_run_*.json` file, a folder containing them, or a
SQLite store (`deepeval.db`, written when `DEEPEVAL_LOCAL_STORE=sqlite`).
A run inside a DB is addressed as `deepeval.db#<run_id>`; without the
suffix (or with `--run-id`) the latest run is opened.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich import print


_INSTALL_HINT = (
    "[bold red]deepeval inspect[/bold red] requires extras that are not "
    "installed.\n"
    "Install them with:\n\n"
    "    pip install 'deepeval\\[inspect]'\n"
)


def inspect_command(
    path: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to a specific test_run_*.json file, a deepeval.db SQLite "
            "store (optionally `deepeval.db#<run_id>`), OR a folder "
            "containing either. If omitted, opens the latest run — from "
            "--folder / DEEPEVAL_RESULTS_FOLDER if set, else the SQLite "
            "store or rolling JSON snapshot deepeval writes after every eval."
        ),
    ),
    folder: Optional[str] = typer.Option(
        None,
        "-f",
        "--folder",
        help=(
            "Folder to scan for the latest test run. Overrides "
            "DEEPEVAL_RESULTS_FOLDER. Ignored when PATH points at a "
            "specific file."
        ),
    ),
    run_id: Optional[int] = typer.Option(
        None,
        "--run-id",
        help=(
            "Open this run id from the SQLite store instead of the latest. "
            "Only meaningful when the resolved source is a deepeval.db."
        ),
    ),
    list_runs: bool = typer.Option(
        False,
        "--list",
        help="List runs stored in the resolved SQLite store and exit.",
    ),
) -> None:
    """Open a TUI to inspect a saved test run's traces.

    Resolution order: PATH (file / `db#id`) → PATH (dir, latest inside) →
    --folder → DEEPEVAL_RESULTS_FOLDER → `.deepeval/deepeval.db` (when
    `DEEPEVAL_LOCAL_STORE=sqlite`) → `.deepeval/.latest_run_full.json`
    (rolling JSON snapshot) → `./experiments`.
    """

    target = _resolve_target(path, folder, run_id)
    if target is None:
        raise typer.BadParameter(
            "No saved test run found. Run an eval first, or pass a "
            "path / folder argument, or set DEEPEVAL_RESULTS_FOLDER."
        )

    if list_runs:
        _print_run_list(target)
        return

    from deepeval.inspect import run_inspect

    try:
        run_inspect(target)
    except ImportError as e:
        # `run_inspect` imports Textual / pyperclip lazily, so a missing
        # extra surfaces here, at call time. Catch any ImportError, not just
        # `textual` -- pyperclip's native bindings can fail late on some
        # platforms.
        print(_INSTALL_HINT)
        print(f"[dim]Underlying error: {e}[/dim]")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        # `find_latest_test_run` can hit this if the folder vanished
        # between resolution and load.
        print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    except Exception as e:
        from deepeval.inspect.loader import InspectLoadError, NoTracesError

        if isinstance(e, (InspectLoadError, NoTracesError)):
            print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)
        raise


def _print_run_list(target: str) -> None:
    from deepeval import sqlite_store

    parsed = sqlite_store.parse_source(target)
    if parsed is None:
        raise typer.BadParameter(
            "--list only works with a SQLite store (deepeval.db). "
            f"Resolved source was: {target}"
        )
    db_path, _ = parsed
    runs = sqlite_store.list_test_runs(db_path, limit=50)
    if not runs:
        print(f"[dim]{db_path} contains no test runs yet.[/dim]")
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title=str(db_path))
    table.add_column("id", justify="right")
    table.add_column("created_at (UTC)")
    table.add_column("identifier")
    table.add_column("passed", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("duration (s)", justify="right")
    table.add_column("cost (USD)", justify="right")
    for r in runs:
        table.add_row(
            str(r["id"]),
            (r.get("created_at") or "")[:19].replace("T", " "),
            r.get("identifier") or "",
            "" if r.get("test_passed") is None else str(r["test_passed"]),
            "" if r.get("test_failed") is None else str(r["test_failed"]),
            (
                ""
                if r.get("run_duration") is None
                else f"{r['run_duration']:.2f}"
            ),
            (
                ""
                if r.get("evaluation_cost") is None
                else f"{r['evaluation_cost']:.4f}"
            ),
        )
    Console().print(table)
    print(
        "[dim]Open one with: [bold]deepeval inspect "
        f"{db_path} --run-id <id>[/bold][/dim]"
    )


def _resolve_target(
    path: Optional[str],
    folder_opt: Optional[str],
    run_id: Optional[int],
) -> Optional[str]:
    from deepeval import sqlite_store
    from deepeval.evaluate.local_store import (
        LOCAL_STORE_SQLITE,
        resolve_local_store_mode,
    )

    if path is not None:
        parsed = sqlite_store.parse_source(path)
        if parsed is not None:
            db_path, embedded_id = parsed
            if not db_path.is_file():
                raise typer.BadParameter(
                    f"SQLite store not found: {db_path}", param_hint="PATH"
                )
            return _db_source(db_path, run_id or embedded_id)

        p = Path(path)
        if p.is_file():
            return str(p)
        if p.is_dir():
            return _find_latest(p, run_id)
        raise typer.BadParameter(
            f"Path not found: {path}",
            param_hint="PATH",
        )

    folder = folder_opt or os.getenv("DEEPEVAL_RESULTS_FOLDER")
    if folder:
        folder_path = Path(folder)
        if folder_path.is_dir():
            return _find_latest(folder_path, run_id)
        return None

    default_db = sqlite_store.resolve_db_path()
    if resolve_local_store_mode() == LOCAL_STORE_SQLITE or run_id is not None:
        if default_db.is_file():
            return _db_source(default_db, run_id)

    from deepeval.test_run.test_run import LATEST_FULL_TEST_RUN_FILE_PATH

    rolling = Path(LATEST_FULL_TEST_RUN_FILE_PATH)
    if rolling.is_file():
        return str(rolling)

    # A DB written earlier in sqlite mode is still the best fallback even
    # if the current process is back in json mode.
    if default_db.is_file():
        return _db_source(default_db, run_id)

    legacy = Path("experiments")
    if legacy.is_dir():
        return _find_latest(legacy, run_id)
    return None


def _db_source(db_path: Path, run_id: Optional[int]) -> str:
    from deepeval import sqlite_store

    if run_id is None:
        return str(db_path)
    return sqlite_store.format_source(db_path, run_id)


def _find_latest(folder: Path, run_id: Optional[int]) -> Optional[str]:
    """Latest run inside `folder`.

    Prefers the SQLite store when the user is in sqlite mode, asked for a
    run id, or the folder has no JSON exports; otherwise the newest
    `test_run_*.json` (existing behaviour).
    """
    from deepeval import sqlite_store
    from deepeval.evaluate.local_store import (
        LOCAL_STORE_SQLITE,
        resolve_local_store_mode,
    )
    from deepeval.inspect.loader import find_latest_test_run

    db_path = folder / sqlite_store.DB_FILENAME
    has_db = db_path.is_file()
    prefer_db = has_db and (
        run_id is not None
        or resolve_local_store_mode() == LOCAL_STORE_SQLITE
    )
    if prefer_db:
        return _db_source(db_path, run_id)

    try:
        return str(find_latest_test_run(folder))
    except FileNotFoundError:
        if has_db:
            return _db_source(db_path, run_id)
        return None
