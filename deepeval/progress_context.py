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
    console = Console(file=sys.stderr)
    with Progress(
        SpinnerColumn(),
        BarColumn(bar_width=60),
        TextColumn("[progress.description]{task.description}"),
        console=console,
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
    long_description: bool = False,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
    pbar_total: Optional[int] = None,
) -> Generator[Tuple[Progress, int], None, None]:
    with capture_synthesizer_run(
        method, max_generations, num_evolutions, evolutions
    ):
        if progress is not None and pbar_id is not None:
            yield progress, pbar_id
        else:
            description = f"✨ Generating up to {max_generations} goldens (method={method}, evolutions={num_evolutions})"
            if long_description:
                if embedder is None:
                    description += (
                        f", using {evaluation_model}, async={async_mode}"
                    )
                else:
                    description += f", using {evaluation_model} and {embedder}, async={async_mode}"
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(bar_width=60),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=custom_console,
            )
            pbar_id = progress.add_task(
                description=description,
                total=pbar_total if pbar_total else max_generations,
            )
            yield progress, pbar_id


@contextmanager
def conversation_simulator_progress_context(
    simulator_model: str,
    num_conversations: int,
    async_mode: bool = False,
    long_description: bool = False,
    progress: Optional[Progress] = None,
    pbar_id: Optional[int] = None,
) -> Generator[Tuple[Progress, int], None, None]:
    with capture_conversation_simulator_run(num_conversations):
        if progress is not None and pbar_id is not None:
            yield progress, pbar_id
        else:
            description = (
                f"🪄 Simulating {num_conversations} conversational test case(s)"
            )
            if long_description:
                description += f"(using {simulator_model}, async={async_mode})"
            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(bar_width=60),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=custom_console,
            )
            pbar_id = progress.add_task(
                description=description, total=num_conversations
            )
            yield progress, pbar_id
