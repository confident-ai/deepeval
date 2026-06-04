"""`deepeval inspect [PATH]` Typer command.

Heavy imports (Textual, pyperclip) are deferred until invocation so
`deepeval.cli.main` stays cheap and users without the optional extra
get a clean install hint instead of a cryptic ImportError.
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
    path: Optional[Path] = typer.Argument(
        None,
        help=(
            "Path to a specific test_run_*.json file, OR a folder "
            "containing them. If omitted, opens the latest run — either "
            "from --folder / DEEPEVAL_RESULTS_FOLDER if set, or the "
            "rolling snapshot deepeval writes after every eval."
        ),
        exists=False,
    ),
    folder: Optional[str] = typer.Option(
        None,
        "-f",
        "--folder",
        help=(
            "Folder to scan for the latest test_run_*.json. Overrides "
            "DEEPEVAL_RESULTS_FOLDER. Ignored when PATH points at a "
            "specific file."
        ),
    ),
) -> None:
    """Open a TUI to inspect a saved test run's traces.

    Resolution order: PATH (file) → PATH (dir, latest inside) → --folder
    → DEEPEVAL_RESULTS_FOLDER → `.deepeval/.latest_run_full.json` (rolling
    snapshot deepeval writes on every eval) → `./experiments`.
    """

    target = _resolve_target(path, folder)
    if target is None:
        raise typer.BadParameter(
            "No test_run_*.json file found. Run an eval first, or pass a "
            "path / folder argument, or set DEEPEVAL_RESULTS_FOLDER."
        )

    # Lazy import so the install hint surfaces before Textual's heavy
    # imports try to load. Catch any ImportError, not just `textual` —
    # pyperclip's native bindings can fail late on some platforms.
    try:
        from deepeval.inspect import run_inspect
    except ImportError as e:
        print(_INSTALL_HINT)
        print(f"[dim]Underlying error: {e}[/dim]")
        raise typer.Exit(code=1)

    try:
        run_inspect(str(target))
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


def _resolve_target(
    path: Optional[Path], folder_opt: Optional[str]
) -> Optional[Path]:
    if path is not None:
        if path.is_file():
            return path
        if path.is_dir():
            return _find_latest(path)
        raise typer.BadParameter(
            f"Path not found: {path}",
            param_hint="PATH",
        )

    folder = folder_opt or os.getenv("DEEPEVAL_RESULTS_FOLDER")
    if folder:
        folder_path = Path(folder)
        if folder_path.is_dir():
            return _find_latest(folder_path)
        return None

    from deepeval.test_run.test_run import LATEST_FULL_TEST_RUN_FILE_PATH

    rolling = Path(LATEST_FULL_TEST_RUN_FILE_PATH)
    if rolling.is_file():
        return rolling

    legacy = Path("experiments")
    if legacy.is_dir():
        return _find_latest(legacy)
    return None


def _find_latest(folder: Path) -> Optional[Path]:
    from deepeval.inspect.loader import find_latest_test_run

    try:
        return find_latest_test_run(folder)
    except FileNotFoundError:
        return None
