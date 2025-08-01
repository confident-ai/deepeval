from typing import Optional, List, Dict
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import ArenaTestCase
from deepeval.metrics import ArenaGEval

from deepeval.utils import get_or_create_event_loop
from deepeval.telemetry import capture_evaluation_run
from collections import Counter


def compare(
    test_cases: List[ArenaTestCase],
    metric: ArenaGEval,
    # Configs
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
) -> Dict[str, int]:
    """
    Compare arena test cases and aggregate winners.

    Args:
        test_cases: List of ArenaTestCase objects to evaluate
        metric: ArenaGEval metric to use for evaluation
        async_config: Configuration for async execution
        display_config: Configuration for display options
        error_config: Configuration for error handling

    Returns:
        Dictionary mapping contestant names to their win counts
    """
    with capture_evaluation_run("compare()"):
        if async_config.run_async:
            loop = get_or_create_event_loop()
            winners = loop.run_until_complete(
                a_compare_arena_test_cases(
                    test_cases=test_cases,
                    metric=metric,
                    ignore_errors=error_config.ignore_errors,
                    verbose_mode=display_config.verbose_mode,
                    show_indicator=display_config.show_indicator,
                    throttle_value=async_config.throttle_value,
                    max_concurrent=async_config.max_concurrent,
                )
            )
        else:
            winners = compare_arena_test_cases(
                test_cases=test_cases,
                metric=metric,
                ignore_errors=error_config.ignore_errors,
                verbose_mode=display_config.verbose_mode,
                show_indicator=display_config.show_indicator,
            )

    # Aggregate winners
    winner_counts = Counter()
    for winner in winners:
        if winner:
            winner_counts[winner] += 1

    return dict(winner_counts)


async def a_compare_arena_test_cases(
    test_cases: List[ArenaTestCase],
    metric: ArenaGEval,
    ignore_errors: bool,
    verbose_mode: bool,
    show_indicator: bool,
    throttle_value: int,
    max_concurrent: int,
) -> List[str]:
    """
    Async version of comparing arena test cases.
    """
    import asyncio

    winners = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_single_test_case(test_case: ArenaTestCase, index: int):
        async with semaphore:
            try:
                # Create a copy of the metric for this test case to avoid conflicts
                metric_copy = ArenaGEval(
                    name=metric.name,
                    evaluation_params=metric.evaluation_params,
                    criteria=metric.criteria,
                    evaluation_steps=metric.evaluation_steps,
                    model=metric.model,
                    async_mode=True,
                    verbose_mode=metric.verbose_mode,
                )

                winner = await metric_copy.a_measure(
                    test_case, _show_indicator=show_indicator
                )
                return winner

            except Exception as e:
                if ignore_errors:
                    if verbose_mode:
                        print(
                            f"Error evaluating test case {index + 1}: {str(e)}"
                        )
                    return None
                else:
                    raise e

    # Create tasks for all test cases
    tasks = [
        evaluate_single_test_case(test_case, i)
        for i, test_case in enumerate(test_cases)
    ]

    # Execute with throttling
    if throttle_value > 0:
        for i in range(0, len(tasks), throttle_value):
            batch = tasks[i : i + throttle_value]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            winners.extend([r for r in batch_results if r is not None])
            if i + throttle_value < len(tasks):
                await asyncio.sleep(0.1)  # Small delay between batches
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        winners = [r for r in results if r is not None]

    return winners


def compare_arena_test_cases(
    test_cases: List[ArenaTestCase],
    metric: ArenaGEval,
    ignore_errors: bool,
    verbose_mode: bool,
    show_indicator: bool,
) -> List[str]:
    """
    Non-async version of comparing arena test cases.
    """
    winners = []

    for i, test_case in enumerate(test_cases):
        try:
            # Create a copy of the metric for this test case to avoid conflicts
            metric_copy = ArenaGEval(
                name=metric.name,
                evaluation_params=metric.evaluation_params,
                criteria=metric.criteria,
                evaluation_steps=metric.evaluation_steps,
                model=metric.model,
                async_mode=False,
                verbose_mode=metric.verbose_mode,
            )

            winner = metric_copy.measure(
                test_case, _show_indicator=show_indicator
            )
            winners.append(winner)

        except Exception as e:
            if ignore_errors:
                if verbose_mode:
                    print(f"Error evaluating test case {i + 1}: {str(e)}")
                continue
            else:
                raise e

    return winners
