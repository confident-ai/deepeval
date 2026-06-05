"""Post-run prompt offering `deepeval inspect` for the saved test run.

Gated so it never fires unexpectedly. The prompt only appears when ALL of:

    1. ``DisplayConfig.inspect_after_run`` is True (the default; set False
       to disable for a single call).
    2. The ``DEEPEVAL_NO_INSPECT_PROMPT`` env var is not truthy
       (escape hatch for CI / non-interactive scripts that own their own
       stdin and shouldn't be hijacked).
    3. ``sys.stdout.isatty()`` — skip in pipes, file redirects, and most
       CI runners. A no-TTY ``input()`` would block forever or raise
       EOFError when the parent process exits.
    4. ``TestRunManager.last_saved_path`` points to an existing file
       (defends against read-only envs and hidden-dir write failures).
    5. The run has at least one ``LLMApiTestCase.trace`` set.
       ``deepeval inspect`` is a trace-tree TUI; runs that only produced
       metric scores would render an empty viewer (and ``load_test_run``
       raises ``NoTracesError`` in that case anyway).

If any gate fails the function returns silently so callers can invoke it
unconditionally at the end of every evaluation pathway.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from deepeval.evaluate.configs import DisplayConfig
    from deepeval.test_run.test_run import TestRunManager


_TRUTHY = {"1", "true", "yes", "on"}
_DISABLE_ENV_VAR = "DEEPEVAL_NO_INSPECT_PROMPT"
_INSTALL_HINT = (
    "[bold red]deepeval inspect[/bold red] requires extras that are not "
    "installed.\nInstall them with:\n\n"
    "    pip install 'deepeval\\[inspect]'\n"
)


def maybe_offer_inspect_tui(
    test_run_manager: "TestRunManager",
    display_config: "DisplayConfig",
) -> None:
    """Offer to open the saved test run in the inspect TUI.

    Safe to call at the end of every ``evaluate()`` / ``evals_iterator()``
    pathway — all gating happens here.
    """
    if not _should_prompt(test_run_manager, display_config):
        return

    saved_path: Path = test_run_manager.last_saved_path  # type: ignore[assignment]
    console = Console()

    try:
        answer = (
            console.input(
                "\n[bold cyan]→[/bold cyan] Open run in "
                "[bold]deepeval inspect[/bold] TUI? [Y/n]: "
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        # User aborted the prompt (^C or stdin closed mid-prompt). Drop a
        # newline so the cursor lands in a clean column and return — we
        # explicitly do NOT propagate KeyboardInterrupt because the eval
        # itself already finished successfully.
        console.print()
        return

    if answer in {"n", "no"}:
        console.print(
            f"[dim]→ You can inspect later with: "
            f"[bold]deepeval inspect {saved_path}[/bold][/dim]"
        )
        return

    try:
        from deepeval.inspect import run_inspect

        run_inspect(str(saved_path))
    except ImportError as e:
        # ``deepeval.inspect.__init__`` defers the heavy Textual / pyperclip
        # imports until ``run_inspect`` is invoked, so the ImportError fires
        # at the call site rather than the import line — catch both via this
        # wider try block.
        console.print(_INSTALL_HINT)
        console.print(f"[dim]Underlying error: {e}[/dim]")
    except Exception as e:
        from deepeval.inspect.loader import InspectLoadError, NoTracesError

        if isinstance(e, (InspectLoadError, NoTracesError)):
            console.print(f"[red]{e}[/red]")
            return
        raise


def _should_prompt(
    test_run_manager: "TestRunManager",
    display_config: "DisplayConfig",
) -> bool:
    if not getattr(display_config, "inspect_after_run", True):
        return False
    if os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY:
        return False
    if not sys.stdout.isatty():
        return False

    saved_path = getattr(test_run_manager, "last_saved_path", None)
    if saved_path is None:
        return False
    try:
        if not Path(saved_path).is_file():
            return False
    except OSError:
        return False

    test_run = getattr(test_run_manager, "test_run", None)
    if test_run is None:
        return False
    cases = getattr(test_run, "test_cases", None) or []
    if not any(getattr(tc, "trace", None) is not None for tc in cases):
        return False

    return True
