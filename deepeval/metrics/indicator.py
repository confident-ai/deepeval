from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from contextlib import contextmanager
import sys
from typing import List, Optional, Union
import time
import asyncio

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
    BaseArenaMetric,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase, MLLMTestCase
from deepeval.test_run.cache import CachedTestCase, Cache
from deepeval.telemetry import capture_metric_type
from deepeval.utils import update_pbar

import logging

logger = logging.getLogger(__name__)


def format_metric_description(
    metric: Union[BaseMetric, BaseConversationalMetric, BaseArenaMetric],
    async_mode: Optional[bool] = None,
):
    if async_mode is None:
        run_async = metric.async_mode
    else:
        run_async = async_mode

    if isinstance(metric, BaseArenaMetric):
        return f"✨ You're running DeepEval's latest [rgb(106,0,255)]{metric.__name__} Metric[/rgb(106,0,255)]! [rgb(55,65,81)](using {metric.evaluation_model}, async_mode={run_async})...[/rgb(55,65,81)]"
    else:
        return f"✨ You're running DeepEval's latest [rgb(106,0,255)]{metric.__name__} Metric[/rgb(106,0,255)]! [rgb(55,65,81)](using {metric.evaluation_model}, strict={metric.strict_mode}, async_mode={run_async})...[/rgb(55,65,81)]"


@contextmanager
def metric_progress_indicator(
    metric: BaseMetric,
    async_mode: Optional[bool] = None,
    total: int = 9999,
    transient: bool = True,
    _show_indicator: bool = True,
    _in_component: bool = False,
):
    captured_async_mode = False if async_mode is None else async_mode
    with capture_metric_type(
        metric.__name__,
        async_mode=captured_async_mode,
        in_component=_in_component,
    ):
        console = Console(file=sys.stderr)  # Direct output to standard error
        if _show_indicator:
            with Progress(
                SpinnerColumn(style="rgb(106,0,255)"),
                BarColumn(bar_width=60),
                TextColumn("[progress.description]{task.description}"),
                console=console,  # Use the custom console
                transient=transient,
            ) as progress:
                progress.add_task(
                    description=format_metric_description(metric, async_mode),
                    total=total,
                )
                yield
        else:
            yield


async def measure_metric_task(
    task_id,
    progress,
    metric: Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric],
    test_case: Union[LLMTestCase, MLLMTestCase, ConversationalTestCase],
    cached_test_case: Union[CachedTestCase, None],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    _in_component: bool = False,
):
    while not progress.finished:
        start_time = time.perf_counter()
        metric_data = None
        if cached_test_case is not None:
            # cached test case will always be None for conversational test case (from a_execute_test_cases)
            cached_metric_data = Cache.get_metric_data(metric, cached_test_case)
            if cached_metric_data:
                metric_data = cached_metric_data.metric_data

        if metric_data:
            ## only change metric state, not configs
            metric.score = metric_data.score
            metric.success = metric_data.success
            metric.reason = metric_data.reason
            metric.evaluation_cost = metric_data.evaluation_cost
            metric.verbose_logs = metric_data.verbose_logs
            finish_text = "Read from Cache"
        else:
            try:
                await metric.a_measure(
                    test_case,
                    _show_indicator=False,
                    _in_component=_in_component,
                    _log_metric_to_confident=False,
                )
                finish_text = "Done"
            except MissingTestCaseParamsError as e:
                if skip_on_missing_params:
                    metric.skipped = True
                    return
                else:
                    if ignore_errors:
                        metric.error = str(e)
                        metric.success = False  # Override metric success
                        finish_text = "Errored"
                    else:
                        raise
            except TypeError:
                try:
                    await metric.a_measure(
                        test_case,
                        _in_component=_in_component,
                        _log_metric_to_confident=False,
                    )
                    finish_text = "Done"
                except MissingTestCaseParamsError as e:
                    if skip_on_missing_params:
                        metric.skipped = True
                        return
                    else:
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
        break


async def measure_metrics_with_indicator(
    metrics: List[
        Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric]
    ],
    test_case: Union[LLMTestCase, MLLMTestCase, ConversationalTestCase],
    cached_test_case: Union[CachedTestCase, None],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    progress: Optional[Progress] = None,
    pbar_eval_id: Optional[int] = None,
    _in_component: bool = False,
):
    if show_indicator:
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            BarColumn(bar_width=60),
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
                        skip_on_missing_params,
                        _in_component=_in_component,
                    )
                )
            await asyncio.gather(*tasks)
    else:
        tasks = []
        for metric in metrics:
            metric_data = None
            # cached test case will always be None for conversationals
            if cached_test_case is not None:
                cached_metric_data = Cache.get_metric_data(
                    metric, cached_test_case
                )
                if cached_metric_data:
                    metric_data = cached_metric_data.metric_data

            if metric_data:
                ## Here we're setting the metric state from metrics metadata cache,
                ## and later using the metric state to create a new metrics metadata cache
                ## WARNING: Potential for bugs, what will happen if a metric changes state in between
                ## test cases?
                metric.score = metric_data.score
                metric.threshold = metric_data.threshold
                metric.success = metric_data.success
                metric.reason = metric_data.reason
                metric.strict_mode = metric_data.strict_mode
                metric.evaluation_model = metric_data.evaluation_model
                metric.evaluation_cost = metric_data.evaluation_cost
                metric.verbose_logs = metric_data.verbose_logs
            else:
                tasks.append(
                    safe_a_measure(
                        metric,
                        test_case,
                        ignore_errors,
                        skip_on_missing_params,
                        progress=progress,
                        pbar_eval_id=pbar_eval_id,
                        _in_component=_in_component,
                    )
                )

        await asyncio.gather(*tasks)


async def safe_a_measure(
    metric: Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric],
    tc: Union[LLMTestCase, MLLMTestCase, ConversationalTestCase],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    progress: Optional[Progress] = None,
    pbar_eval_id: Optional[int] = None,
    _in_component: bool = False,
):
    try:
        await metric.a_measure(
            tc,
            _show_indicator=False,
            _in_component=_in_component,
            _log_metric_to_confident=False,
        )
        update_pbar(progress, pbar_eval_id)

    except asyncio.CancelledError:
        logger.info("caught asyncio.CancelledError")

        # treat cancellation as a timeout so we still emit a MetricData
        metric.error = (
            "Timed out/cancelled while evaluating metric. "
            "Increase DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE or set "
            "DEEPEVAL_LOG_STACK_TRACES=1 for full traceback."
        )
        metric.success = False

        if not ignore_errors:
            raise

    except MissingTestCaseParamsError as e:
        if skip_on_missing_params:
            metric.skipped = True
            return
        else:
            if ignore_errors:
                metric.error = str(e)
                metric.success = False
            else:
                raise
    except TypeError:
        try:
            await metric.a_measure(tc)
        except MissingTestCaseParamsError as e:
            if skip_on_missing_params:
                metric.skipped = True
                return
            else:
                if ignore_errors:
                    metric.error = str(e)
                    metric.success = False
                else:
                    raise
    except Exception as e:
        if ignore_errors:
            metric.error = str(e)
            metric.success = False  # Assuming you want to set success to False
            logger.info("a metric was marked as errored")
        else:
            raise
