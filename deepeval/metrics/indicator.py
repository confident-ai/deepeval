import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from contextlib import contextmanager
import sys
from typing import List, Optional
import time

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.utils import show_indicator


def format_metric_description(
    metric: BaseMetric, is_async: Optional[bool] = None
):
    if is_async is None:
        run_async = metric.run_async
    else:
        run_async = is_async

    return f"✨ You're running DeepEval's latest [rgb(106,0,255)]{metric.__name__} Metric[/rgb(106,0,255)]! [rgb(55,65,81)](using {metric.evaluation_model}, strict={metric.strict_mode})...[/rgb(55,65,81)]"


@contextmanager
def metric_progress_indicator(
    metric: BaseMetric,
    is_async: Optional[bool] = None,
    _show_indicator: bool = True,
    total: int = 9999,
    transient: bool = True,
):
    console = Console(file=sys.stderr)  # Direct output to standard error
    if _show_indicator and show_indicator():
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            console=console,  # Use the custom console
            transient=transient,
        ) as progress:
            progress.add_task(
                description=format_metric_description(metric, is_async),
                total=total,
            )
            yield
    else:
        yield


async def measure_metric_task(
    task_id, progress, metric: BaseMetric, test_case: LLMTestCase
):
    while not progress.finished:
        start_time = time.perf_counter()
        await metric.a_measure(test_case, _show_indicator=False)
        end_time = time.perf_counter()
        time_taken = format(end_time - start_time, ".2f")
        progress.update(task_id, advance=100)
        progress.update(
            task_id,
            description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done! ({time_taken}s)",
        )
        break


async def measure_metrics_with_indicator(
    metrics: List[BaseMetric],
    test_case: LLMTestCase,
):
    if show_indicator():
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            tasks = []
            for metric in metrics:
                task_id = progress.add_task(
                    description=format_metric_description(metric), total=100
                )
                tasks.append(
                    measure_metric_task(task_id, progress, metric, test_case)
                )
            await asyncio.gather(*tasks)
    else:
        await asyncio.gather(
            *[
                metric.a_measure(test_case, _show_indicator=False)
                for metric in metrics
            ]
        )
