from typing import Optional, List, Dict, Callable
import asyncio
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from collections import Counter

from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import ArenaTestCase
from deepeval.metrics import ArenaGEval
from deepeval.utils import add_pbar, update_pbar, custom_console
from deepeval.utils import get_or_create_event_loop
from deepeval.telemetry import capture_evaluation_run


def compare(
    test_cases: List[ArenaTestCase],
    metric: ArenaGEval,
    # Configs
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
) -> Dict[str, int]:
    with capture_evaluation_run("compare()"):
        if async_config.run_async:
            loop = get_or_create_event_loop()
            winners = loop.run_until_complete(
                a_execute_arena_test_cases(
                    test_cases=test_cases,
                    metric=metric,
                    ignore_errors=error_config.ignore_errors,
                    verbose_mode=display_config.verbose_mode,
                    show_indicator=display_config.show_indicator,
                    throttle_value=async_config.throttle_value,
                    max_concurrent=async_config.max_concurrent,
                    skip_on_missing_params=error_config.skip_on_missing_params,
                )
            )
        else:
            winners = execute_arena_test_cases(
                test_cases=test_cases,
                metric=metric,
                ignore_errors=error_config.ignore_errors,
                verbose_mode=display_config.verbose_mode,
                show_indicator=display_config.show_indicator,
                skip_on_missing_params=error_config.skip_on_missing_params,
            )

    # Aggregate winners
    winner_counts = Counter()
    for winner in winners:
        if winner:
            winner_counts[winner] += 1

    print(winner_counts)
    return dict(winner_counts)


async def a_execute_arena_test_cases(
    test_cases: List[ArenaTestCase],
    metric: ArenaGEval,
    ignore_errors: bool,
    verbose_mode: bool,
    show_indicator: bool,
    throttle_value: int,
    skip_on_missing_params: bool,
    max_concurrent: int,
) -> List[str]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    winners = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_single_test_case(
        test_case: ArenaTestCase,
        index: int,
        progress: Optional[Progress] = None,
        pbar_id: Optional[int] = None,
    ):
        pbar_test_case_id = add_pbar(
            progress,
            f"    🧐 Picking a winner (#{index + 1})",
            total=3,
        )
        metric_copy = ArenaGEval(
            name=metric.name,
            evaluation_params=metric.evaluation_params,
            criteria=metric.criteria,
            evaluation_steps=metric.evaluation_steps,
            model=metric.model,
            async_mode=False,
            verbose_mode=(
                verbose_mode
                if verbose_mode is not None
                else metric.verbose_mode
            ),
        )
        winner = await _a_handle_metric_measurement(
            metric=metric_copy,
            test_case=test_case,
            ignore_errors=ignore_errors,
            skip_on_missing_params=skip_on_missing_params,
            _progress=progress,
            _pbar_id=pbar_test_case_id,
        )
        if winner:
            winners.append(winner)

        update_pbar(progress, pbar_id)

    # Create tasks for all test cases
    if show_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        with progress:
            pbar_id = add_pbar(
                progress,
                f"🆚 Comparing {len(test_cases)} contestants concurrently",
                total=len(test_cases),
            )
            tasks = []
            for i, test_case in enumerate(test_cases):
                task = execute_with_semaphore(
                    func=evaluate_single_test_case,
                    test_case=test_case,
                    progress=progress,
                    pbar_id=pbar_id,
                    index=i,
                )
                tasks.append(asyncio.create_task(task))
                await asyncio.sleep(throttle_value)

            await asyncio.gather(*tasks)

    return winners


def execute_arena_test_cases(
    test_cases: List[ArenaTestCase],
    metric: ArenaGEval,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    verbose_mode: Optional[bool] = None,
) -> List[str]:
    """
    Non-async version of comparing arena test cases.
    """
    winners = []

    # TODO: doesn't work
    def evaluate_test_cases(progress=None, pbar_id=None):
        for i, test_case in enumerate(test_cases):
            pbar_test_case_id = add_pbar(
                progress,
                f"    🧐 Picking a winner (#{i + 1})",
                total=3,
            )
            metric_copy = ArenaGEval(
                name=metric.name,
                evaluation_params=metric.evaluation_params,
                criteria=metric.criteria,
                evaluation_steps=metric.evaluation_steps,
                model=metric.model,
                async_mode=False,
                verbose_mode=(
                    verbose_mode
                    if verbose_mode is not None
                    else metric.verbose_mode
                ),
            )
            winner = _handle_metric_measurement(
                metric=metric_copy,
                test_case=test_case,
                ignore_errors=ignore_errors,
                skip_on_missing_params=skip_on_missing_params,
                _progress=progress,
                _pbar_id=pbar_test_case_id,
            )
            if winner:
                winners.append(winner)

            update_pbar(progress, pbar_id)

    if show_indicator:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=60),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=custom_console,
        )
        with progress:
            pbar_id = add_pbar(
                progress,
                f"🆚 Comparing {len(test_cases)} contestants sequentially",
                total=len(test_cases),
            )
            evaluate_test_cases(progress=progress, pbar_id=pbar_id)
    else:
        evaluate_test_cases()

    return winners


def _handle_metric_measurement(
    metric: ArenaGEval,
    test_case: ArenaTestCase,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    _progress: Optional[Progress] = None,
    _pbar_id: Optional[int] = None,
) -> Optional[str]:
    try:
        winner = metric.measure(
            test_case,
            _show_indicator=False,
            _progress=_progress,
            _pbar_id=_pbar_id,
        )
        return winner
    except MissingTestCaseParamsError as e:
        if skip_on_missing_params:
            return None
        else:
            if ignore_errors:
                metric.error = str(e)
                metric.success = False
                return None
            else:
                raise
    except TypeError:
        try:
            winner = metric.measure(test_case)
            return winner
        except MissingTestCaseParamsError as e:
            if skip_on_missing_params:
                return None
            else:
                if ignore_errors:
                    metric.error = str(e)
                    metric.success = False
                    return None
                else:
                    raise
        except Exception as e:
            if ignore_errors:
                metric.error = str(e)
                metric.success = False
                return None
            else:
                raise


async def _a_handle_metric_measurement(
    metric: ArenaGEval,
    test_case: ArenaTestCase,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    _progress: Optional[Progress] = None,
    _pbar_id: Optional[int] = None,
) -> Optional[str]:
    try:
        winner = await metric.a_measure(
            test_case,
            _show_indicator=False,
            _progress=_progress,
            _pbar_id=_pbar_id,
        )
        return winner
    except MissingTestCaseParamsError as e:
        if skip_on_missing_params:
            return None
        else:
            if ignore_errors:
                metric.error = str(e)
                metric.success = False
                return None
            else:
                raise
    except TypeError:
        try:
            winner = await metric.a_measure(test_case)
            return winner
        except MissingTestCaseParamsError as e:
            if skip_on_missing_params:
                return None
            else:
                if ignore_errors:
                    metric.error = str(e)
                    metric.success = False
                    return None
                else:
                    raise
        except Exception as e:
            if ignore_errors:
                metric.error = str(e)
                metric.success = False
                return None
            else:
                raise
