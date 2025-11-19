from typing import Optional, List, Dict, Callable
import asyncio
import time
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from collections import Counter
import json

from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import ArenaTestCase, Contestant
from deepeval.test_case.api import create_api_test_case
from deepeval.metrics import ArenaGEval
from deepeval.utils import (
    add_pbar,
    update_pbar,
    custom_console,
    get_or_create_event_loop,
    open_browser,
)
from deepeval.test_run.test_run import (
    TestRun,
    MetricData,
    TestRunEncoder,
    MetricScores,
    console,
)
from deepeval.test_run.hyperparameters import (
    process_hyperparameters,
)
from deepeval.confident.api import Api, Endpoints, HttpMethods, is_confident
from deepeval.telemetry import capture_evaluation_run
from deepeval.test_run.api import LLMApiTestCase
from deepeval.evaluate.utils import create_arena_metric_data
from deepeval.evaluate.types import PostExperimentRequest


def compare(
    test_cases: List[ArenaTestCase],
    metric: ArenaGEval,
    name: str = "compare()",
    # Configs
    async_config: Optional[AsyncConfig] = AsyncConfig(),
    display_config: Optional[DisplayConfig] = DisplayConfig(),
    error_config: Optional[ErrorConfig] = ErrorConfig(),
) -> Dict[str, int]:

    # Prepare test run map
    unique_contestant_names = set(
        [
            contestant.name
            for test_case in test_cases
            for contestant in test_case.contestants
        ]
    )
    test_run_map: Dict[str, TestRun] = {}
    for contestant_name in unique_contestant_names:
        test_run = TestRun(
            identifier=contestant_name,
            test_passed=0,
            test_failed=0,
        )
        test_run.metrics_scores = [
            MetricScores(
                metric=metric.name,
                scores=[],
                passes=0,
                fails=0,
                errors=0,
            )
        ]
        test_run_map[contestant_name] = test_run

    start_time = time.time()
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
                    test_run_map=test_run_map,
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
                test_run_map=test_run_map,
            )
    end_time = time.time()
    run_duration = end_time - start_time

    # Aggregate winners
    winner_counts = Counter()
    for winner in winners:
        if winner:
            winner_counts[winner] += 1

    process_test_runs(test_run_map=test_run_map, test_cases=test_cases)
    wrap_up_experiment(
        name=name,
        test_runs=list(test_run_map.values()),
        winner_counts=winner_counts,
        run_duration=run_duration,
    )
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
    test_run_map: Dict[str, TestRun],
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

        start_time = time.perf_counter()
        winner = await _a_handle_metric_measurement(
            metric=metric_copy,
            test_case=test_case,
            ignore_errors=ignore_errors,
            skip_on_missing_params=skip_on_missing_params,
            _progress=progress,
            _pbar_id=pbar_test_case_id,
        )
        end_time = time.perf_counter()
        run_duration = end_time - start_time

        if winner:
            winners.append(winner)

        update_pbar(progress, pbar_id)
        update_test_run_map(
            test_case=test_case,
            index=index,
            test_run_map=test_run_map,
            metric_copy=metric_copy,
            winner=winner,
            run_duration=run_duration,
        )

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
    test_run_map: Optional[Dict[str, TestRun]] = None,
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

            start_time = time.perf_counter()
            winner = _handle_metric_measurement(
                metric=metric_copy,
                test_case=test_case,
                ignore_errors=ignore_errors,
                skip_on_missing_params=skip_on_missing_params,
                _progress=progress,
                _pbar_id=pbar_test_case_id,
            )
            end_time = time.perf_counter()
            run_duration = end_time - start_time

            if winner:
                winners.append(winner)

            update_pbar(progress, pbar_id)
            update_test_run_map(
                test_case=test_case,
                index=i,
                test_run_map=test_run_map,
                metric_copy=metric_copy,
                winner=winner,
                run_duration=run_duration,
            )

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


def update_test_run_map(
    test_case: ArenaTestCase,
    index: int,
    test_run_map: Dict[str, TestRun],
    metric_copy: ArenaGEval,
    winner: str,
    run_duration: float,
):
    for contestant in test_case.contestants:
        test_run = test_run_map.get(contestant.name)

        # update test cases in test run
        api_test_case: LLMApiTestCase = create_api_test_case(
            test_case=contestant.test_case, index=index
        )
        metric_data: MetricData = create_arena_metric_data(
            metric_copy, contestant.name
        )
        api_test_case.update_metric_data(metric_data)
        api_test_case.update_run_duration(run_duration)
        test_run.add_test_case(api_test_case)

        # update other test run attributes
        if test_run.run_duration is None:
            test_run.run_duration = 0.0
        test_run.run_duration += run_duration

        # Ensure test_passed and test_failed are initialized
        if test_run.test_passed is None:
            test_run.test_passed = 0
        if test_run.test_failed is None:
            test_run.test_failed = 0

        if winner == contestant:
            test_run.test_passed += 1
        else:
            test_run.test_failed += 1

        # update metric scores
        test_run.metrics_scores[0].metric = metric_copy.name
        test_run.metrics_scores[0].scores.append(
            1 if winner == contestant else 0
        )
        test_run.metrics_scores[0].passes += 1 if winner == contestant else 0
        test_run.metrics_scores[0].fails += 1 if winner != contestant else 0
        test_run.metrics_scores[0].errors += 0


def process_test_runs(
    test_run_map: Dict[str, TestRun],
    test_cases: List[ArenaTestCase],
):
    hyperparameters_map = {
        contestant_name: {} for contestant_name in test_run_map.keys()
    }

    for test_case in test_cases:
        for contestant in test_case.contestants:
            if contestant.hyperparameters:
                hyperparameters_map[contestant.name].update(
                    contestant.hyperparameters
                )

    for contestant_name, hyperparameters in hyperparameters_map.items():
        test_run = test_run_map.get(contestant_name)
        test_run.hyperparameters = process_hyperparameters(hyperparameters)


def wrap_up_experiment(
    name: str,
    test_runs: List[TestRun],
    winner_counts: Counter,
    run_duration: float,
):
    winner_breakdown = []
    for contestant, wins in winner_counts.most_common():
        winner_breakdown.append(
            f"    » [bold green]{contestant}[/bold green]: {wins} wins"
        )
    winner_text = (
        "\n".join(winner_breakdown) if winner_breakdown else "No winners"
    )
    console.print(
        f"\n🎉 Arena completed! (time taken: {round(run_duration, 2)}s | token cost: {test_runs[0].evaluation_cost if test_runs else 0} USD)\n"
        f"🏆 Results ({sum(winner_counts.values())} total test cases):\n"
        f"{winner_text}\n\n"
    )

    if not is_confident():
        console.print(
            f"{'=' * 80}\n"
            f"\n» Want to share experiments with your team? ❤️ 🏟️\n"
            f"  » Run [bold]'deepeval login'[/bold] to analyze and save arena results on [rgb(106,0,255)]Confident AI[/rgb(106,0,255)].\n\n"
        )
        return

    try:
        api = Api()
        experiment_request = PostExperimentRequest(
            testRuns=test_runs, name=name
        )

        try:
            body = experiment_request.model_dump(
                by_alias=True, exclude_none=True
            )
        except AttributeError:
            body = experiment_request.dict(by_alias=True, exclude_none=True)
        json_str = json.dumps(body, cls=TestRunEncoder)
        body = json.loads(json_str)

        _, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.EXPERIMENT_ENDPOINT,
            body=body,
        )
        console.print(
            "[rgb(5,245,141)]✓[/rgb(5,245,141)] Done 🎉! View results on "
            f"[link={link}]{link}[/link]"
        )
        open_browser(link)

    except Exception:
        raise
