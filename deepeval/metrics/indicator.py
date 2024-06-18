from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from contextlib import contextmanager
import sys
from typing import List, Optional, Union
import time
import asyncio
import threading
import contextvars

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.utils import (
    show_indicator,
    capture_contextvars,
    update_contextvars,
)
from deepeval.test_run.cache import CachedTestCase, Cache
from deepeval.telemetry import capture_metric_type


def format_metric_description(
    metric: BaseMetric, async_mode: Optional[bool] = None
):
    if async_mode is None:
        run_async = metric.async_mode
    else:
        run_async = async_mode

    if run_async:
        is_async = "yes"
    else:
        is_async = "no"

    return f"✨ You're running DeepEval's latest [rgb(106,0,255)]{metric.__name__} Metric[/rgb(106,0,255)]! [rgb(55,65,81)](using {metric.evaluation_model}, strict={metric.strict_mode}, async_mode={run_async})...[/rgb(55,65,81)]"


# Initialize console and progress with the console
console = Console(file=sys.stderr)
progress = Progress(
    SpinnerColumn(style="rgb(106,0,255)"),
    TextColumn("[progress.description]{task.description}"),
    console=console,
    transient=True,
)

# A lock to safely update task count
# lock = threading.Lock()
# active_tasks = 0

# def start_progress():
#     with lock:
#         global active_tasks
#         if active_tasks == 0:
#             progress.start()
#         active_tasks += 1

# def update_progress():
#     with lock:
#         global active_tasks
#         active_tasks -= 1
#         if active_tasks == 0:
#             progress.stop()


@contextmanager
def metric_progress_indicator(
    metric, async_mode=False, _show_indicator=True, total=9999
):
    with capture_metric_type(metric.__name__):
        if _show_indicator and show_indicator():
            # if async_mode==False:
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                TextColumn("[progress.description]{task.description}"),
                console=console,  # Use the custom console
                transient=True,
            ) as progress_sync:
                progress_sync.add_task(
                    description=format_metric_description(metric, async_mode),
                    total=total,
                )
                yield
            # else:
            #     start_progress()
            #     start_time = time.perf_counter()
            #     task_id = progress.add_task(
            #         description=format_metric_description(metric, async_mode),
            #         total=total,
            #     )
            #     try:
            #         yield task_id
            #     finally:
            #         end_time = time.perf_counter()
            #         time_taken = format(end_time - start_time, ".2f")
            #         progress.update(task_id, completed=total)
            #         progress.update(
            #             task_id,
            #             description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done! ({time_taken}s)",
            #         )
            #         progress.stop_task(task_id)
            #         update_progress()
        else:
            yield None


async def measure_metric_task(
    task_id,
    progress,
    metric: BaseMetric,
    test_case: Union[LLMTestCase, ConversationalTestCase],
    cached_test_case: Union[CachedTestCase, None],
    ignore_errors: bool,
    metric_states: dict,
):
    while not progress.finished:
        start_time = time.perf_counter()
        metric_metadata = None
        if cached_test_case is not None:
            # cached test casr will always be None for conversational test case (from a_execute_test_cases)
            cached_metric_data = Cache.get_metric_data(metric, cached_test_case)
            if cached_metric_data:
                metric_metadata = cached_metric_data.metric_metadata

        if metric_metadata:
            ## only change metric state, not configs
            metric.score = metric_metadata.score
            metric.success = metric_metadata.success
            metric.reason = metric_metadata.reason
            metric.evaluation_cost = metric_metadata.evaluation_cost
            finish_text = "Read from Cache"
        else:
            if isinstance(test_case, ConversationalTestCase):
                tc = test_case.messages[len(test_case.messages) - 1]
            else:
                tc = test_case

            try:
                await metric.a_measure(tc, _show_indicator=False)
                print(metric.score, metric.reason)
                finish_text = "Done"
            except TypeError:
                try:
                    await metric.a_measure(tc)
                    finish_text = "Done"
                except Exception as e:
                    if ignore_errors:
                        metric.error = str(e)
                        metric.success = False  # Override metric success
                        finish_text = "Errored"
                    else:
                        raise
            except Exception as e:
                if ignore_errors:
                    metric.error = str(e)
                    metric.success = False  # Override metric success
                    finish_text = "Errored"
                else:
                    raise

        end_time = time.perf_counter()
        time_taken = format(end_time - start_time, ".2f")
        progress.update(task_id, advance=100)
        progress.update(
            task_id,
            description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]{finish_text}! ({time_taken}s)",
        )

        context_vars = capture_contextvars(metric)
        metric_states[metric] = context_vars

        break


async def measure_metrics_with_indicator(
    metrics: List[BaseMetric],
    test_case: Union[LLMTestCase, ConversationalTestCase],
    cached_test_case: Union[CachedTestCase, None],
    ignore_errors: bool,
):
    metric_states = {}

    if show_indicator():
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            tasks = []
            for metric in metrics:
                task_id = progress.add_task(
                    description=format_metric_description(
                        metric, async_mode=True
                    ),
                    total=100,
                )
                tasks.append(
                    measure_metric_task(
                        task_id,
                        progress,
                        metric,
                        test_case,
                        cached_test_case,
                        ignore_errors,
                        metric_states,
                    )
                )
            await asyncio.gather(*tasks)

    else:
        tasks = []
        for metric in metrics:
            metric_metadata = None
            # cached test case will always be None for conversationals
            if cached_test_case is not None:
                cached_metric_data = Cache.get_metric_data(
                    metric, cached_test_case
                )
                if cached_metric_data:
                    metric_metadata = cached_metric_data.metric_metadata

            if metric_metadata:
                ## Here we're setting the metric state from metrics metadata cache,
                ## and later using the metric state to create a new metrics metadata cache
                ## WARNING: Potential for bugs, what will happen if a metric changes state in between
                ## test cases?
                metric.score = metric_metadata.score
                metric.threshold = metric_metadata.threshold
                metric.success = metric_metadata.success
                metric.reason = metric_metadata.reason
                metric.strict_mode = metric_metadata.strict_mode
                metric.evaluation_model = metric_metadata.evaluation_model
                metric.evaluation_cost = metric_metadata.evaluation_cost
            else:
                if isinstance(test_case, ConversationalTestCase):
                    tc = test_case.messages[len(test_case.messages) - 1]
                else:
                    tc = test_case

                tasks.append(safe_a_measure(metric, tc, ignore_errors))

        await asyncio.gather(*tasks)

    # Update the metrics with the states captured from the tasks
    for metric in metrics:
        context_vars = metric_states[metric]
        update_contextvars(metric, context_vars)


async def safe_a_measure(
    metric: BaseMetric, tc: LLMTestCase, ignore_errors: bool
):
    try:
        try:
            await metric.a_measure(tc, _show_indicator=False)
        except TypeError:
            await metric.a_measure(tc)
    except Exception as e:
        if ignore_errors:
            metric.error = str(e)
            metric.success = False  # Assuming you want to set success to False
        else:
            raise
