import asyncio
from copy import deepcopy
import os
from typing import Callable, List, Optional, Union, Dict
import time
from dataclasses import dataclass
from pydantic import BaseModel
from rich.console import Console
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.utils import copy_metrics
from deepeval.test_case.utils import check_valid_test_cases_type
from deepeval.test_run.hyperparameters import process_hyperparameters
from deepeval.test_run.test_run import TestRunResultDisplay
from deepeval.utils import (
    get_or_create_event_loop,
    should_ignore_errors,
    should_skip_on_missing_params,
    should_use_cache,
    should_verbose_print,
)
from deepeval.telemetry import capture_evaluation_run
from deepeval.metrics import (
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
)
from deepeval.metrics.indicator import (
    format_metric_description,
    measure_metrics_with_indicator,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
    MLLMImage,
)
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import (
    global_test_run_manager,
    LLMApiTestCase,
    ConversationalApiTestCase,
    MetricData,
    TestRunManager,
    TestRun,
)
from deepeval.utils import get_is_running_deepeval
from deepeval.test_run.cache import (
    global_test_run_cache_manager,
    Cache,
    CachedTestCase,
    CachedMetricData,
)


@dataclass
class TestResult:
    """Returned from run_test"""

    name: str
    success: bool
    metrics_data: Union[List[MetricData], None]
    conversational: bool
    multimodal: Optional[bool] = None
    input: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    actual_output: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None


class EvaluationResult(BaseModel):
    test_results: List[TestResult]
    confident_link: Optional[str]


def create_metric_data(metric: BaseMetric) -> MetricData:
    if metric.error is not None:
        return MetricData(
            name=metric.__name__,
            threshold=metric.threshold,
            score=None,
            reason=None,
            success=False,
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=metric.error,
            evaluationCost=metric.evaluation_cost,
            verboseLogs=metric.verbose_logs,
        )
    else:
        return MetricData(
            name=metric.__name__,
            score=metric.score,
            threshold=metric.threshold,
            reason=metric.reason,
            success=metric.is_successful(),
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=None,
            evaluationCost=metric.evaluation_cost,
            verboseLogs=metric.verbose_logs,
        )


def create_test_result(
    api_test_case: Union[LLMApiTestCase, ConversationalApiTestCase],
) -> TestResult:
    name = api_test_case.name

    if isinstance(api_test_case, ConversationalApiTestCase):
        return TestResult(
            name=name,
            success=api_test_case.success,
            metrics_data=api_test_case.metrics_data,
            conversational=True,
        )
    else:
        multimodal = (
            api_test_case.multimodal_input is not None
            and api_test_case.multimodal_input_actual_output is not None
        )
        if multimodal:
            return TestResult(
                name=name,
                success=api_test_case.success,
                metrics_data=api_test_case.metrics_data,
                input=api_test_case.multimodal_input,
                actual_output=api_test_case.multimodal_input_actual_output,
                conversational=False,
                multimodal=True,
            )
        else:
            return TestResult(
                name=name,
                success=api_test_case.success,
                metrics_data=api_test_case.metrics_data,
                input=api_test_case.input,
                actual_output=api_test_case.actual_output,
                expected_output=api_test_case.expected_output,
                context=api_test_case.context,
                retrieval_context=api_test_case.retrieval_context,
                conversational=False,
                multimodal=False,
            )


def create_api_test_case(
    test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    index: Optional[int] = None,
    conversational_instance_id: Optional[int] = None,
    additional_metadata: Optional[Dict] = None,
    comments: Optional[str] = None,
) -> Union[LLMApiTestCase, ConversationalApiTestCase]:
    if isinstance(test_case, ConversationalTestCase):
        if test_case.name:
            name = test_case.name
        else:
            order = (
                test_case._dataset_rank
                if test_case._dataset_rank is not None
                else index
            )

            name = os.getenv(
                PYTEST_RUN_TEST_NAME, f"conversational_test_case_{order}"
            )

        api_test_case = ConversationalApiTestCase(
            name=name,
            success=True,
            metricsData=[],
            runDuration=0,
            evaluationCost=None,
            order=order,
            testCases=[],
            additionalMetadata=test_case.additional_metadata,
        )
        api_test_case.instance_id = id(api_test_case)
        api_test_case.turns = [
            create_api_test_case(
                turn,
                index,
                api_test_case.instance_id,
                test_case.additional_metadata,
                test_case.comments,
            )
            for index, turn in enumerate(test_case.turns)
        ]

        return api_test_case
    else:
        if conversational_instance_id:
            success = None
            name = f"turn_{index}"
            order = index

            # Manually set the metadata and comments on conversational test case
            # to each individual message (test case)
            test_case.additional_metadata = additional_metadata
            test_case.comments = comments
            metrics_data = None
        else:
            order = (
                test_case._dataset_rank
                if test_case._dataset_rank is not None
                else index
            )

            success = True
            if test_case.name is not None:
                name = test_case.name
            else:
                name = os.getenv(PYTEST_RUN_TEST_NAME, f"test_case_{order}")
            metrics_data = []

        if isinstance(test_case, LLMTestCase):
            api_test_case = LLMApiTestCase(
                name=name,
                input=test_case.input,
                actualOutput=test_case.actual_output,
                expectedOutput=test_case.expected_output,
                context=test_case.context,
                retrievalContext=test_case.retrieval_context,
                toolsCalled=test_case.tools_called,
                expectedTools=test_case.expected_tools,
                success=success,
                metricsData=metrics_data,
                runDuration=None,
                evaluationCost=None,
                order=order,
                additionalMetadata=test_case.additional_metadata,
                comments=test_case.comments,
                conversational_instance_id=conversational_instance_id,
            )
        elif isinstance(test_case, MLLMTestCase):
            api_test_case = LLMApiTestCase(
                name=name,
                multimodalInput=test_case.input,
                multimodalActualOutput=test_case.actual_output,
                toolsCalled=test_case.tools_called,
                expectedTools=test_case.expected_tools,
                success=success,
                metricsData=metrics_data,
                runDuration=None,
                evaluationCost=None,
                order=order,
                additionalMetadata=test_case.additional_metadata,
                comments=test_case.comments,
                conversational_instance_id=conversational_instance_id,
            )

        # llm_test_case_lookup_map[instance_id] = api_test_case
        return api_test_case


def execute_test_cases(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase, MLLMTestCase]],
    metrics: List[
        Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric]
    ],
    skip_on_missing_params: bool,
    ignore_errors: bool,
    use_cache: bool,
    show_indicator: bool,
    save_to_disk: bool = False,
    verbose_mode: Optional[bool] = None,
    identifier: Optional[str] = None,
    test_run_manager: Optional[TestRunManager] = None,
    _use_bar_indicator: bool = True,
) -> List[TestResult]:
    global_test_run_cache_manager.disable_write_cache = save_to_disk == False

    if test_run_manager is None:
        test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run(identifier=identifier)

    if verbose_mode is not None:
        for metric in metrics:
            metric.verbose_mode = verbose_mode

    conversational_metrics: List[BaseConversationalMetric] = []
    llm_metrics: List[BaseMetric] = []
    mllm_metrics: List[BaseMultimodalMetric] = []
    for metric in metrics:
        metric.async_mode = False
        if isinstance(metric, BaseMetric):
            llm_metrics.append(metric)
        elif isinstance(metric, BaseConversationalMetric):
            conversational_metrics.append(metric)
        elif isinstance(metric, BaseMultimodalMetric):
            mllm_metrics.append(metric)

    test_results: List[TestResult] = []

    def evaluate_test_cases(pbar: Optional[tqdm] = None):
        llm_test_case_count = -1
        conversational_test_case_count = -1
        show_metric_indicator = show_indicator and not _use_bar_indicator
        for test_case in test_cases:
            with capture_evaluation_run("test case"):
                for metric in metrics:
                    metric.error = None  # Reset metric error

                if isinstance(test_case, LLMTestCase):
                    if len(llm_metrics) == 0:
                        continue

                    llm_test_case_count += 1
                    cached_test_case = None
                    if use_cache:
                        cached_test_case = (
                            global_test_run_cache_manager.get_cached_test_case(
                                test_case, test_run.hyperparameters
                            )
                        )

                    ##### Metric Calculation #####
                    api_test_case: LLMApiTestCase = create_api_test_case(
                        test_case, llm_test_case_count
                    )
                    new_cached_test_case: CachedTestCase = CachedTestCase()

                    test_start_time = time.perf_counter()
                    read_all_metrics_from_cache = True
                    for metric in llm_metrics:
                        metric_data = None
                        if cached_test_case is not None:
                            cached_metric_data = Cache.get_metric_data(
                                metric, cached_test_case
                            )
                            if cached_metric_data:
                                metric_data = cached_metric_data.metric_data

                        if metric_data is None:
                            read_all_metrics_from_cache = False
                            try:
                                metric.measure(
                                    test_case,
                                    _show_indicator=show_metric_indicator,
                                )
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except TypeError:
                                try:
                                    metric.measure(test_case)
                                except MissingTestCaseParamsError as e:
                                    if skip_on_missing_params:
                                        continue
                                    else:
                                        if ignore_errors:
                                            metric.error = str(e)
                                            metric.success = False
                                        else:
                                            raise
                                except Exception as e:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                            metric_data = create_metric_data(metric)

                        # here, we will check for an additional property on the flattened test cases to see if updating is necessary
                        api_test_case.update_metric_data(metric_data)
                        if metric.error is None:
                            cache_metric_data = deepcopy(metric_data)
                            cache_metric_data.evaluation_cost = 0  # Cached metrics will have evaluation cost as 0, not None.
                            updated_cached_metric_data = CachedMetricData(
                                metric_data=cache_metric_data,
                                metric_configuration=Cache.create_metric_configuration(
                                    metric
                                ),
                            )
                            new_cached_test_case.cached_metrics_data.append(
                                updated_cached_metric_data
                            )

                    test_end_time = time.perf_counter()
                    if read_all_metrics_from_cache:
                        run_duration = 0
                    else:
                        run_duration = test_end_time - test_start_time
                    api_test_case.update_run_duration(run_duration)

                    ### Update Test Run ###
                    test_run_manager.update_test_run(api_test_case, test_case)

                    ### Cache Test Run ###
                    global_test_run_cache_manager.cache_test_case(
                        test_case,
                        new_cached_test_case,
                        test_run.hyperparameters,
                    )
                    global_test_run_cache_manager.cache_test_case(
                        test_case,
                        new_cached_test_case,
                        test_run.hyperparameters,
                        to_temp=True,
                    )

                # No caching and not sending test cases to Confident AI for multimodal metrics yet
                elif isinstance(test_case, MLLMTestCase):
                    if len(mllm_metrics) == 0:
                        continue

                    api_test_case: LLMApiTestCase = create_api_test_case(
                        test_case, llm_test_case_count
                    )
                    test_start_time = time.perf_counter()
                    for metric in mllm_metrics:
                        try:
                            metric.measure(
                                test_case,
                                _show_indicator=show_metric_indicator,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(test_case)
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise
                        metric_data = create_metric_data(metric)
                        api_test_case.update_metric_data(metric_data)

                    test_end_time = time.perf_counter()
                    if len(mllm_metrics) > 0:
                        run_duration = test_end_time - test_start_time
                        api_test_case.update_run_duration(run_duration)

                    ### Update Test Run ###
                    test_run_manager.update_test_run(api_test_case, test_case)

                # No caching for conversational metrics yet
                elif isinstance(test_case, ConversationalTestCase):
                    if len(metrics) == 0:
                        continue

                    conversational_test_case_count += 1
                    api_test_case: ConversationalApiTestCase = (
                        create_api_test_case(
                            test_case, conversational_test_case_count
                        )
                    )

                    test_start_time = time.perf_counter()
                    for metric in metrics:
                        try:
                            metric.measure(
                                test_case,
                                _show_indicator=show_metric_indicator,
                            )
                        except MissingTestCaseParamsError as e:
                            if skip_on_missing_params:
                                continue
                            else:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except TypeError:
                            try:
                                metric.measure(test_case)
                            except MissingTestCaseParamsError as e:
                                if skip_on_missing_params:
                                    continue
                                else:
                                    if ignore_errors:
                                        metric.error = str(e)
                                        metric.success = False
                                    else:
                                        raise
                            except Exception as e:
                                if ignore_errors:
                                    metric.error = str(e)
                                    metric.success = False
                                else:
                                    raise
                        except Exception as e:
                            if ignore_errors:
                                metric.error = str(e)
                                metric.success = False
                            else:
                                raise

                        metric_data = create_metric_data(metric)
                        api_test_case.update_metric_data(metric_data)

                    test_end_time = time.perf_counter()
                    run_duration = test_end_time - test_start_time
                    api_test_case.update_run_duration(run_duration)

                    ### Update Test Run ###
                    test_run_manager.update_test_run(api_test_case, test_case)

                test_result = create_test_result(api_test_case)
                test_results.append(test_result)

                if pbar is not None:
                    pbar.update(1)

    if show_indicator and _use_bar_indicator:
        with tqdm(
            desc=f"Evaluating {len(test_cases)} test case(s) sequentially",
            unit="test case",
            total=len(test_cases),
            bar_format="{desc}: |{bar}|{percentage:3.0f}% ({n_fmt}/{total_fmt}) [Time Taken: {elapsed}, {rate_fmt}{postfix}]",
        ) as pbar:
            evaluate_test_cases(pbar)
    else:
        evaluate_test_cases()

    return test_results


async def a_execute_test_cases(
    test_cases: List[Union[LLMTestCase, ConversationalTestCase, MLLMTestCase]],
    metrics: List[
        Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric]
    ],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    use_cache: bool,
    show_indicator: bool,
    throttle_value: int,
    max_concurrent: int,
    save_to_disk: bool = False,
    verbose_mode: Optional[bool] = None,
    identifier: Optional[str] = None,
    test_run_manager: Optional[TestRunManager] = None,
    _use_bar_indicator: bool = True,
) -> List[TestResult]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    global_test_run_cache_manager.disable_write_cache = save_to_disk == False
    if test_run_manager is None:
        test_run_manager = global_test_run_manager

    test_run_manager.save_to_disk = save_to_disk
    test_run = test_run_manager.get_test_run(identifier=identifier)

    if verbose_mode is not None:
        for metric in metrics:
            metric.verbose_mode = verbose_mode

    llm_metrics: List[BaseMetric] = []
    mllm_metrics: List[BaseMultimodalMetric] = []
    conversational_metrics: List[BaseConversationalMetric] = []
    for metric in metrics:
        if isinstance(metric, BaseMetric):
            llm_metrics.append(metric)
        elif isinstance(metric, BaseMultimodalMetric):
            mllm_metrics.append(metric)
        elif isinstance(metric, BaseConversationalMetric):
            conversational_metrics.append(metric)

    llm_test_case_counter = -1
    mllm_test_case_counter = -1
    conversational_test_case_counter = -1
    test_results: List[Union[TestResult, MLLMTestCase]] = []
    tasks = []

    if show_indicator and _use_bar_indicator:
        with tqdm_asyncio(
            desc=f"Evaluating {len(test_cases)} test case(s) in parallel",
            unit="test case",
            total=len(test_cases),
            bar_format="{desc}: |{bar}|{percentage:3.0f}% ({n_fmt}/{total_fmt}) [Time Taken: {elapsed}, {rate_fmt}{postfix}]",
        ) as pbar:
            for test_case in test_cases:
                with capture_evaluation_run("test case"):
                    if isinstance(test_case, LLMTestCase):
                        if len(llm_metrics) == 0:
                            pbar.update(1)
                            continue

                        llm_test_case_counter += 1
                        copied_llm_metrics: List[BaseMetric] = copy_metrics(
                            llm_metrics
                        )
                        task = execute_with_semaphore(
                            func=a_execute_llm_test_cases,
                            metrics=copied_llm_metrics,
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=llm_test_case_counter,
                            test_run=test_run,
                            ignore_errors=ignore_errors,
                            skip_on_missing_params=skip_on_missing_params,
                            use_cache=use_cache,
                            show_indicator=show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            pbar=pbar,
                        )
                        tasks.append(asyncio.create_task(task))

                    elif isinstance(test_case, MLLMTestCase):
                        mllm_test_case_counter += 1
                        copied_multimodal_metrics: List[
                            BaseMultimodalMetric
                        ] = copy_metrics(mllm_metrics)
                        task = execute_with_semaphore(
                            func=a_execute_mllm_test_cases,
                            metrics=copied_multimodal_metrics,
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=mllm_test_case_counter,
                            ignore_errors=ignore_errors,
                            skip_on_missing_params=skip_on_missing_params,
                            show_indicator=show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            pbar=pbar,
                        )
                        tasks.append(asyncio.create_task(task))

                    elif isinstance(test_case, ConversationalTestCase):
                        conversational_test_case_counter += 1

                        task = execute_with_semaphore(
                            func=a_execute_conversational_test_cases,
                            metrics=copy_metrics(metrics),
                            test_case=test_case,
                            test_run_manager=test_run_manager,
                            test_results=test_results,
                            count=conversational_test_case_counter,
                            ignore_errors=ignore_errors,
                            skip_on_missing_params=skip_on_missing_params,
                            show_indicator=show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            pbar=pbar,
                        )
                        tasks.append(asyncio.create_task(task))

                    await asyncio.sleep(throttle_value)
            await asyncio.gather(*tasks)
    else:
        for test_case in test_cases:
            with capture_evaluation_run("test case"):
                if isinstance(test_case, LLMTestCase):
                    if len(llm_metrics) == 0:
                        continue
                    llm_test_case_counter += 1

                    copied_llm_metrics: List[BaseMetric] = copy_metrics(
                        llm_metrics
                    )
                    task = execute_with_semaphore(
                        func=a_execute_llm_test_cases,
                        metrics=copied_llm_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=llm_test_case_counter,
                        test_run=test_run,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        use_cache=use_cache,
                        _use_bar_indicator=_use_bar_indicator,
                        show_indicator=show_indicator,
                    )
                    tasks.append(asyncio.create_task((task)))

                elif isinstance(test_case, ConversationalTestCase):
                    conversational_test_case_counter += 1
                    copied_conversational_metrics: List[
                        BaseConversationalMetric
                    ] = []
                    copied_conversational_metrics = copy_metrics(
                        conversational_metrics
                    )
                    task = execute_with_semaphore(
                        func=a_execute_conversational_test_cases,
                        metrics=copied_conversational_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=conversational_test_case_counter,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        _use_bar_indicator=_use_bar_indicator,
                        show_indicator=show_indicator,
                    )
                    tasks.append(asyncio.create_task((task)))

                elif isinstance(test_case, MLLMTestCase):
                    mllm_test_case_counter += 1
                    copied_multimodal_metrics: List[BaseMultimodalMetric] = (
                        copy_metrics(mllm_metrics)
                    )
                    task = execute_with_semaphore(
                        func=a_execute_mllm_test_cases,
                        metrics=copied_multimodal_metrics,
                        test_case=test_case,
                        test_run_manager=test_run_manager,
                        test_results=test_results,
                        count=mllm_test_case_counter,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        _use_bar_indicator=_use_bar_indicator,
                        show_indicator=show_indicator,
                    )
                    tasks.append(asyncio.create_task(task))

                await asyncio.sleep(throttle_value)
        await asyncio.gather(*tasks)

    return test_results


async def a_execute_llm_test_cases(
    metrics: List[BaseMetric],
    test_case: LLMTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    test_run: TestRun,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    use_cache: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    pbar: Optional[tqdm_asyncio] = None,
):
    show_metrics_indicator = show_indicator and not _use_bar_indicator

    cached_test_case = None
    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error

    # only use cache when NOT conversational test case
    if use_cache:
        cached_test_case = global_test_run_cache_manager.get_cached_test_case(
            test_case,
            test_run.hyperparameters,
        )

    ##### Metric Calculation #####
    api_test_case = create_api_test_case(test_case, count)
    new_cached_test_case: CachedTestCase = CachedTestCase()
    test_start_time = time.perf_counter()
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=cached_test_case,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
    )

    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        api_test_case.update_metric_data(metric_data)

        if metric.error is None:
            cache_metric_data = deepcopy(metric_data)
            cache_metric_data.evaluation_cost = (
                0  # Create new copy and save 0 for cost
            )
            updated_cached_metric_data = CachedMetricData(
                metric_data=cache_metric_data,
                metric_configuration=Cache.create_metric_configuration(metric),
            )
            new_cached_test_case.cached_metrics_data.append(
                updated_cached_metric_data
            )

    test_end_time = time.perf_counter()
    run_duration = test_end_time - test_start_time
    # Quick hack to check if all metrics were from cache
    if run_duration < 1:
        run_duration = 0
    api_test_case.update_run_duration(run_duration)

    ### Update Test Run ###
    test_run_manager.update_test_run(api_test_case, test_case)

    ### Cache Test Run ###
    global_test_run_cache_manager.cache_test_case(
        test_case,
        new_cached_test_case,
        test_run.hyperparameters,
    )
    global_test_run_cache_manager.cache_test_case(
        test_case,
        new_cached_test_case,
        test_run.hyperparameters,
        to_temp=True,
    )

    test_results.append(create_test_result(api_test_case))

    if pbar is not None:
        pbar.update(1)


async def a_execute_mllm_test_cases(
    metrics: List[BaseMultimodalMetric],
    test_case: MLLMTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    pbar: Optional[tqdm_asyncio] = None,
):
    show_metrics_indicator = show_indicator and not _use_bar_indicator

    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error

    api_test_case: LLMApiTestCase = create_api_test_case(test_case, count)
    test_start_time = time.perf_counter()
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=None,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
    )
    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        api_test_case.update_metric_data(metric_data)

    test_end_time = time.perf_counter()
    run_duration = test_end_time - test_start_time
    api_test_case.update_run_duration(run_duration)

    ### Update Test Run ###
    test_run_manager.update_test_run(api_test_case, test_case)
    test_results.append(create_test_result(api_test_case))

    if pbar is not None:
        pbar.update(1)


async def a_execute_conversational_test_cases(
    metrics: List[
        Union[BaseMetric, BaseMultimodalMetric, BaseConversationalMetric]
    ],
    test_case: ConversationalTestCase,
    test_run_manager: TestRunManager,
    test_results: List[Union[TestResult, MLLMTestCase]],
    count: int,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    pbar: Optional[tqdm_asyncio] = None,
):
    show_metrics_indicator = show_indicator and not _use_bar_indicator

    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error

    api_test_case: ConversationalApiTestCase = create_api_test_case(
        test_case, count
    )

    test_start_time = time.perf_counter()
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=None,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
    )
    for metric in metrics:
        if metric.skipped:
            continue

        metric_data = create_metric_data(metric)
        api_test_case.update_metric_data(metric_data)

    test_end_time = time.perf_counter()
    if len(metrics) > 0:
        run_duration = test_end_time - test_start_time
        api_test_case.update_run_duration(run_duration)

    ### Update Test Run ###
    test_run_manager.update_test_run(api_test_case, test_case)

    test_results.append(create_test_result(api_test_case))

    if pbar is not None:
        pbar.update(1)


def assert_test(
    test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    metrics: List[
        Union[BaseMetric, BaseConversationalMetric, BaseMultimodalMetric]
    ],
    run_async: bool = True,
):
    if run_async:
        loop = get_or_create_event_loop()
        test_result = loop.run_until_complete(
            a_execute_test_cases(
                [test_case],
                metrics,
                skip_on_missing_params=should_skip_on_missing_params(),
                ignore_errors=should_ignore_errors(),
                use_cache=should_use_cache(),
                verbose_mode=should_verbose_print(),
                throttle_value=0,
                # this doesn't matter for pytest
                max_concurrent=100,
                save_to_disk=get_is_running_deepeval(),
                show_indicator=True,
                _use_bar_indicator=True,
            )
        )[0]
    else:
        test_result = execute_test_cases(
            [test_case],
            metrics,
            skip_on_missing_params=should_skip_on_missing_params(),
            ignore_errors=should_ignore_errors(),
            use_cache=should_use_cache(),
            verbose_mode=should_verbose_print(),
            save_to_disk=get_is_running_deepeval(),
            show_indicator=True,
            _use_bar_indicator=False,
        )[0]

    if not test_result.success:
        failed_metrics_data: List[MetricData] = []
        # even for conversations, test_result right now is just the
        # result for the last message
        for metric_data in test_result.metrics_data:
            if metric_data.error is not None:
                failed_metrics_data.append(metric_data)
            else:
                # This try block is for user defined custom metrics,
                # which might not handle the score == undefined case elegantly
                try:
                    if not metric_data.success:
                        failed_metrics_data.append(metric_data)
                except:
                    failed_metrics_data.append(metric_data)

        failed_metrics_str = ", ".join(
            [
                f"{metrics_data.name} (score: {metrics_data.score}, threshold: {metrics_data.threshold}, strict: {metrics_data.strict_mode}, error: {metrics_data.error})"
                for metrics_data in failed_metrics_data
            ]
        )
        raise AssertionError(f"Metrics: {failed_metrics_str} failed.")


def evaluate(
    test_cases: Union[
        List[Union[LLMTestCase, MLLMTestCase]], List[ConversationalTestCase]
    ],
    metrics: List[BaseMetric],
    hyperparameters: Optional[Dict[str, Union[str, int, float]]] = None,
    run_async: bool = True,
    show_indicator: bool = True,
    print_results: bool = True,
    write_cache: bool = True,
    use_cache: bool = False,
    ignore_errors: bool = False,
    skip_on_missing_params: bool = False,
    verbose_mode: Optional[bool] = None,
    identifier: Optional[str] = None,
    throttle_value: int = 0,
    max_concurrent: int = 100,
    display: Optional[TestRunResultDisplay] = TestRunResultDisplay.ALL,
) -> EvaluationResult:
    check_valid_test_cases_type(test_cases)

    if hyperparameters is not None:
        if (
            hyperparameters.get("model") is None
            or hyperparameters.get("prompt template") is None
        ):
            raise ValueError(
                "A `model` and `prompt template` key must be provided when logging `hyperparameters`."
            )
        hyperparameters = process_hyperparameters(hyperparameters)

    global_test_run_manager.reset()
    start_time = time.perf_counter()

    if show_indicator:
        console = Console()
        for metric in metrics:
            console.print(
                format_metric_description(metric, async_mode=run_async)
            )

    with capture_evaluation_run("evaluate()"):
        if run_async:
            loop = get_or_create_event_loop()
            test_results = loop.run_until_complete(
                a_execute_test_cases(
                    test_cases,
                    metrics,
                    ignore_errors=ignore_errors,
                    use_cache=use_cache,
                    verbose_mode=verbose_mode,
                    save_to_disk=write_cache,
                    show_indicator=show_indicator,
                    skip_on_missing_params=skip_on_missing_params,
                    throttle_value=throttle_value,
                    identifier=identifier,
                    max_concurrent=max_concurrent,
                )
            )
        else:
            test_results = execute_test_cases(
                test_cases,
                metrics,
                ignore_errors=ignore_errors,
                use_cache=use_cache,
                verbose_mode=verbose_mode,
                save_to_disk=write_cache,
                skip_on_missing_params=skip_on_missing_params,
                identifier=identifier,
                show_indicator=show_indicator,
            )

    end_time = time.perf_counter()
    run_duration = end_time - start_time
    if print_results:
        for test_result in test_results:
            print_test_result(test_result, display)

        aggregate_metric_pass_rates(test_results)

    test_run = global_test_run_manager.get_test_run()
    test_run.hyperparameters = hyperparameters
    global_test_run_manager.save_test_run()
    confident_link = global_test_run_manager.wrap_up_test_run(
        run_duration, display_table=False
    )
    return EvaluationResult(
        test_results=test_results, confident_link=confident_link
    )


def print_test_result(test_result: TestResult, display: TestRunResultDisplay):
    if test_result.metrics_data is None:
        return

    if (
        display == TestRunResultDisplay.PASSING.value
        and test_result.success is False
    ):
        return
    elif display == TestRunResultDisplay.FAILING.value and test_result.success:
        return

    print("")
    print("=" * 70 + "\n")
    print("Metrics Summary\n")

    for metric_data in test_result.metrics_data:
        successful = True
        if metric_data.error is not None:
            successful = False
        else:
            # This try block is for user defined custom metrics,
            # which might not handle the score == undefined case elegantly
            try:
                if not metric_data.success:
                    successful = False
            except:
                successful = False

        if not successful:
            print(
                f"  - ❌ {metric_data.name} (score: {metric_data.score}, threshold: {metric_data.threshold}, strict: {metric_data.strict_mode}, evaluation model: {metric_data.evaluation_model}, reason: {metric_data.reason}, error: {metric_data.error})"
            )
        else:
            print(
                f"  - ✅ {metric_data.name} (score: {metric_data.score}, threshold: {metric_data.threshold}, strict: {metric_data.strict_mode}, evaluation model: {metric_data.evaluation_model}, reason: {metric_data.reason}, error: {metric_data.error})"
            )

    print("")
    if test_result.multimodal:
        print("For multimodal test case:\n")
        print(f"  - input: {test_result.input}")
        print(f"  - actual output: {test_result.actual_output}")

    elif test_result.conversational:
        print("For conversational test case:\n")
        print(
            f"  - Unable to print conversational test case. Login to Confident AI (https://app.confident-ai.com) to view conversational evaluations in full."
        )
    else:
        print("For test case:\n")
        print(f"  - input: {test_result.input}")
        print(f"  - actual output: {test_result.actual_output}")
        print(f"  - expected output: {test_result.expected_output}")
        print(f"  - context: {test_result.context}")
        print(f"  - retrieval context: {test_result.retrieval_context}")


def aggregate_metric_pass_rates(test_results: List[TestResult]) -> dict:
    metric_counts = {}
    metric_successes = {}

    for result in test_results:
        if result.metrics_data:
            for metric_data in result.metrics_data:
                metric_name = metric_data.name
                if metric_name not in metric_counts:
                    metric_counts[metric_name] = 0
                    metric_successes[metric_name] = 0
                metric_counts[metric_name] += 1
                if metric_data.success:
                    metric_successes[metric_name] += 1

    metric_pass_rates = {
        metric: (metric_successes[metric] / metric_counts[metric])
        for metric in metric_counts
    }

    print("\n" + "=" * 70 + "\n")
    print("Overall Metric Pass Rates\n")
    for metric, pass_rate in metric_pass_rates.items():
        print(f"{metric}: {pass_rate:.2%} pass rate")
    print("\n" + "=" * 70 + "\n")

    return metric_pass_rates
