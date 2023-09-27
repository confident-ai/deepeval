from rich.progress import Progress, SpinnerColumn, TextColumn
from contextlib import contextmanager


@contextmanager
def progress_context(
    description: str, total: int = 9999, transient: bool = True
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=transient,
    ) as progress:
        progress.add_task(description=description, total=total)
        yield
