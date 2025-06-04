from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from typing import Optional, Generator
from contextlib import contextmanager
from rich.console import Console
from typing import Dict, Tuple
import sys

from deepeval.telemetry import (
    capture_synthesizer_run,
    capture_conversation_simulator_run,
)
from deepeval.utils import custom_console


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
    async_mode: bool = False,
) -> Generator[str, None, None]:
    with capture_synthesizer_run(
        method, max_generations, num_evolutions, evolutions
    ):
        if embedder is None:
            description = f"✨ Generating up to {max_generations} goldens" # using DeepEval (using {evaluation_model}, method={method}, async={async_mode})
        else:
            description = f"✨ Generating up to {max_generations} goldens" #  using DeepEval (using {evaluation_model} and {embedder}, method={method}, async={async_mode})
        yield description


@contextmanager
def conversation_simulator_progress_context(
    simulator_model: str,
    num_conversations: int,
    async_mode: bool = False,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
    long_description: bool = False,
) -> Generator[Tuple[Progress, int], None, None]:
    with capture_conversation_simulator_run(num_conversations):
        if progress is not None and pbar_id is not None:
            yield progress, pbar_id
        else:
            description = f"🪄 Simulating {num_conversations} conversational test case(s)"
            if long_description:
                description += f"(using {simulator_model}, async={async_mode})"
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=custom_console,
            )
            pbar_id = progress.add_task(description=description, total=num_conversations)
            yield progress, pbar_id
