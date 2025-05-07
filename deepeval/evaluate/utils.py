from typing import Optional, List, Callable, Union, Dict
import os


from deepeval.test_run.test_run import TestRunResultDisplay
from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
    MLLMImage,
)
from deepeval.test_run import (
    LLMApiTestCase,
    ConversationalApiTestCase,
    MetricData,
)
from deepeval.evaluate.types import TestResult
from deepeval.tracing.api import TraceApi
from deepeval.tracing.tracing import BaseSpan, Trace
from deepeval.constants import PYTEST_RUN_TEST_NAME


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
            additional_metadata=api_test_case.additional_metadata,
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
                additional_metadata=api_test_case.additional_metadata,
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
                additional_metadata=api_test_case.additional_metadata,
            )


def create_api_test_case(
    test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    trace: Optional[TraceApi] = None,
    index: Optional[int] = None,
    conversational_instance_id: Optional[int] = None,
    additional_metadata: Optional[Dict] = None,
    comments: Optional[str] = None,
) -> Union[LLMApiTestCase, ConversationalApiTestCase]:
    if isinstance(test_case, ConversationalTestCase):
        order = (
            test_case._dataset_rank
            if test_case._dataset_rank is not None
            else index
        )
        if test_case.name:
            name = test_case.name
        else:
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
                test_case=turn,
                index=index,
                conversational_instance_id=api_test_case.instance_id,
                additional_metadata=turn.additional_metadata,
                comments=turn.comments,
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
                tokenCost=test_case.token_cost,
                completionTime=test_case.completion_time,
                success=success,
                metricsData=metrics_data,
                runDuration=None,
                evaluationCost=None,
                order=order,
                additionalMetadata=test_case.additional_metadata,
                comments=test_case.comments,
                conversational_instance_id=conversational_instance_id,
                trace=trace,
            )
        elif isinstance(test_case, MLLMTestCase):
            api_test_case = LLMApiTestCase(
                name=name,
                multimodalInput=test_case.input,
                multimodalActualOutput=test_case.actual_output,
                toolsCalled=test_case.tools_called,
                expectedTools=test_case.expected_tools,
                tokenCost=test_case.token_cost,
                completionTime=test_case.completion_time,
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


def validate_assert_test_inputs(
    golden: Optional[Golden] = None,
    observed_callback: Optional[Callable] = None,
    test_case: Optional[LLMTestCase] = None,
    metrics: Optional[List] = None,
):
    if golden and observed_callback:
        if not getattr(observed_callback, "_is_deepeval_observed", False):
            raise ValueError(
                "The provided 'observed_callback' must be decorated with '@observe' from deepeval.tracing."
            )
        if test_case or metrics:
            raise ValueError(
                "You cannot provide both ('golden' + 'observed_callback') and ('test_case' + 'metrics'). Choose one mode."
            )
    elif (golden and not observed_callback) or (
        observed_callback and not golden
    ):
        raise ValueError(
            "Both 'golden' and 'observed_callback' must be provided together."
        )

    if (test_case and not metrics) or (metrics and not test_case):
        raise ValueError(
            "Both 'test_case' and 'metrics' must be provided together."
        )

    if not ((golden and observed_callback) or (test_case and metrics)):
        raise ValueError(
            "You must provide either ('golden' + 'observed_callback') or ('test_case' + 'metrics')."
        )


def validate_evaluate_inputs(
    goldens: Optional[List] = None,
    observed_callback: Optional[Callable] = None,
    test_cases: Optional[List] = None,
    metrics: Optional[List] = None,
):

    if goldens and observed_callback:
        if not getattr(observed_callback, "_is_deepeval_observed", False):
            raise ValueError(
                "The provided 'observed_callback' must be decorated with '@observe' from deepeval.tracing."
            )
        if test_cases or metrics:
            raise ValueError(
                "You cannot provide both ('goldens' with 'observed_callback') and ('test_cases' with 'metrics'). Please choose one mode."
            )
    elif (goldens and not observed_callback) or (
        observed_callback and not goldens
    ):
        raise ValueError(
            "If using 'goldens', you must also provide a 'observed_callback'."
        )
    if (test_cases and not metrics) or (metrics and not test_cases):
        raise ValueError(
            "If using 'test_cases', you must also provide 'metrics'."
        )
    if not ((goldens and observed_callback) or (test_cases and metrics)):
        raise ValueError(
            "You must provide either goldens with a 'observed_callback', or test_cases with 'metrics'."
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


def count_metrics_in_trace(trace: Trace) -> int:
    def count_metrics_recursive(span: BaseSpan) -> int:
        count = len(span.metrics) if span.metrics else 0
        for child in span.children:
            count += count_metrics_recursive(child)
        return count

    return sum(count_metrics_recursive(span) for span in trace.root_spans)
