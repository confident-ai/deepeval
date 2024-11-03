from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm.asyncio import tqdm as async_tqdm_bar
from typing import Optional, Generator
from contextlib import contextmanager
from tqdm import tqdm as tqdm_bar
from rich.console import Console
from typing import Dict
import tqdm
import sys

from deepeval.telemetry import capture_synthesizer_run


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
    method: str,
    evaluation_model: str,
    num_evolutions: int,
    evolutions: Dict,
    embedder: Optional[str] = None,
    max_generations: str = None,
    progress_bar: Optional[tqdm.std.tqdm] = None,
    async_mode: bool = False,
) -> Generator[Optional[tqdm.std.tqdm], None, None]:
    with capture_synthesizer_run(
        method, max_generations, num_evolutions, evolutions
    ):
        if embedder is None:
            description = f"✨ Generating up to {max_generations} goldens using DeepEval (using {evaluation_model}, method={method})"
        else:
            description = f"✨ Generating up to {max_generations} goldens using DeepEval (using {evaluation_model} and {embedder}, method={method})"
        # Direct output to stderr, using TQDM progress bar for visual feedback
        if not progress_bar:
            if async_mode:
                with async_tqdm_bar(
                    total=max_generations, desc=description, file=sys.stderr
                ) as progress_bar:
                    yield progress_bar
            else:
                with tqdm_bar(
                    total=max_generations, desc=description, file=sys.stderr
                ) as progress_bar:
                    yield progress_bar
        else:
            yield progress_bar
