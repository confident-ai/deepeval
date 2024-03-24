from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from contextlib import contextmanager
import sys


@contextmanager
def progress_context(
    description: str, total: int = 9999, transient: bool = True
):
    console = Console(file=sys.stderr)  # Direct output to standard error
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,  # Use the custom console
        transient=transient,
    ) as progress:
        progress.add_task(description=description, total=total)
        yield


@contextmanager
def synthesizer_progress_context(
    evaluation_model: str,
    _show_indicator: bool = True,
    total: int = 9999,
    transient: bool = True,
):
    description = f"✨ 🍰 ✨ You're generating goldens using DeepEval's latest Synthesizer (using {evaluation_model})! This may take a while..."
    console = Console(file=sys.stderr)  # Direct output to standard error
    if _show_indicator:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,  # Use the custom console
            transient=transient,
        ) as progress:
            progress.add_task(description=description, total=total)
            yield
    else:
        yield
