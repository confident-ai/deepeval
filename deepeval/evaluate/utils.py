import ast
import inspect
from typing import Optional, List, Callable, Union
import os
import time

from deepeval.utils import format_turn
from deepeval.test_run.test_run import TestRunResultDisplay
from deepeval.dataset import Golden
from deepeval.metrics import (
    ArenaGEval,
    BaseMetric,
    BaseConversationalMetric,
    BaseMultimodalMetric,
)
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    MLLMTestCase,
)
from deepeval.test_run import (
    LLMApiTestCase,
    ConversationalApiTestCase,
    MetricData,
)
from deepeval.evaluate.types import TestResult
from deepeval.tracing.api import TraceApi, BaseApiSpan, TraceSpanApiStatus
from deepeval.tracing.tracing import BaseSpan, Trace
from deepeval.tracing.types import TraceSpanStatus
from deepeval.tracing.utils import (
    perf_counter_to_datetime,
    to_zod_compatible_iso,
)


def _is_metric_successful(metric_data: MetricData) -> bool:
    """
    Robustly determine success for a metric row.

    Rationale:
    - If the metric recorded an error, treat as failure.
    - Be defensive: custom rows may not be MetricData at runtime.
    """
    if getattr(metric_data, "error", None):
        return False

    s = getattr(metric_data, "success", None)
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    if isinstance(s, (int, float)):
        return bool(s)
    if isinstance(s, str):
        return s.strip().lower() in {"true", "t", "1", "yes", "y"}
    return False


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


def create_arena_metric_data(metric: ArenaGEval, contestant: str) -> MetricData:
    if metric.error is not None:
        return MetricData(
            name=metric.__name__,
            threshold=1,
            score=None,
            reason=None,
            success=False,
            strictMode=True,
            evaluationModel=metric.evaluation_model,
            error=metric.error,
            evaluationCost=metric.evaluation_cost,
            verboseLogs=metric.verbose_logs,
        )
    else:
        return MetricData(
            name=metric.__name__,
            score=1 if contestant == metric.winner else 0,
            threshold=1,
            reason=metric.reason,
            success=metric.is_successful(),
            strictMode=True,
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
            turns=api_test_case.turns,
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


def create_api_trace(trace: Trace, golden: Golden) -> TraceApi:
    return TraceApi(
        uuid=trace.uuid,
        baseSpans=[],
        agentSpans=[],
        llmSpans=[],
        retrieverSpans=[],
        toolSpans=[],
        startTime=(
            to_zod_compatible_iso(perf_counter_to_datetime(trace.start_time))
            if trace.start_time
            else None
        ),
        endTime=(
            to_zod_compatible_iso(perf_counter_to_datetime(trace.end_time))
            if trace.end_time
            else None
        ),
        input=trace.input,
        output=trace.output,
        expected_output=trace.expected_output,
        context=trace.context,
        retrieval_context=trace.retrieval_context,
        tools_called=trace.tools_called,
        expected_tools=trace.expected_tools,
        metadata=golden.additional_metadata,
        status=(
            TraceSpanApiStatus.SUCCESS
            if trace.status == TraceSpanStatus.SUCCESS
            else TraceSpanApiStatus.ERRORED
        ),
    )


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

    if test_case and metrics:
        if isinstance(test_case, LLMTestCase) and not all(
            isinstance(metric, BaseMetric) for metric in metrics
        ):
            raise ValueError(
                "All 'metrics' for an 'LLMTestCase' must be instances of 'BaseMetric' only."
            )
        if isinstance(test_case, ConversationalTestCase) and not all(
            isinstance(metric, BaseConversationalMetric) for metric in metrics
        ):
            raise ValueError(
                "All 'metrics' for an 'ConversationalTestCase' must be instances of 'BaseConversationalMetric' only."
            )
        if isinstance(test_case, MLLMTestCase) and not all(
            isinstance(metric, BaseMultimodalMetric) for metric in metrics
        ):
            raise ValueError(
                "All 'metrics' for an 'MLLMTestCase' must be instances of 'BaseMultimodalMetric' only."
            )

    if not ((golden and observed_callback) or (test_case and metrics)):
        raise ValueError(
            "You must provide either ('golden' + 'observed_callback') or ('test_case' + 'metrics')."
        )


def validate_evaluate_inputs(
    goldens: Optional[List] = None,
    observed_callback: Optional[Callable] = None,
    test_cases: Optional[
        Union[
            List[LLMTestCase], List[ConversationalTestCase], List[MLLMTestCase]
        ]
    ] = None,
    metrics: Optional[
        Union[
            List[BaseMetric],
            List[BaseConversationalMetric],
            List[BaseMultimodalMetric],
        ]
    ] = None,
    metric_collection: Optional[str] = None,
):
    if metric_collection is None and metrics is None:
        raise ValueError(
            "You must provide either 'metric_collection' or 'metrics'."
        )
    if metric_collection is not None and metrics is not None:
        raise ValueError(
            "You cannot provide both 'metric_collection' and 'metrics'."
        )

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

    if test_cases and metrics:
        for test_case in test_cases:
            for metric in metrics:
                if isinstance(test_case, LLMTestCase) and not isinstance(
                    metric, BaseMetric
                ):
                    raise ValueError(
                        f"Metric {metric.__name__} is not a valid metric for LLMTestCase."
                    )
                if isinstance(
                    test_case, ConversationalTestCase
                ) and not isinstance(metric, BaseConversationalMetric):
                    print(type(metric))
                    raise ValueError(
                        f"Metric {metric.__name__} is not a valid metric for ConversationalTestCase."
                    )
                if isinstance(test_case, MLLMTestCase) and not isinstance(
                    metric, BaseMultimodalMetric
                ):
                    raise ValueError(
                        f"Metric {metric.__name__} is not a valid metric for MLLMTestCase."
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
        successful = _is_metric_successful(metric_data)

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
        if test_result.turns:
            print("  Turns:")
            turns = sorted(test_result.turns, key=lambda t: t.order)
            for t in turns:
                print(format_turn(t))
        else:
            print("  - No turns recorded in this test case.")

    else:
        print("For test case:\n")
        print(f"  - input: {test_result.input}")
        print(f"  - actual output: {test_result.actual_output}")
        print(f"  - expected output: {test_result.expected_output}")
        print(f"  - context: {test_result.context}")
        print(f"  - retrieval context: {test_result.retrieval_context}")


def write_test_result_to_file(
    test_result: TestResult, display: TestRunResultDisplay, output_dir: str
):

    def get_log_id(output_dir: str):
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output_dir, f"test_run_{ts}.log")
        return log_path

    def aggregate_metric_pass_rates_to_file(test_results: List[TestResult]):
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
        with open(out_file, "a", encoding="utf-8") as file:
            file.write("\n" + "=" * 70 + "\n")
            file.write("Overall Metric Pass Rates\n")
            for metric, pass_rate in metric_pass_rates.items():
                file.write(f"{metric}: {pass_rate:.2%} pass rate")
            file.write("\n" + "=" * 70 + "\n")

    # Determine output Directory
    out_dir = output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    # Generate log id
    out_file = get_log_id(out_dir)

    if test_result.metrics_data is None:
        return

    if (
        display == TestRunResultDisplay.PASSING.value
        and test_result.success is False
    ):
        return
    elif display == TestRunResultDisplay.FAILING.value and test_result.success:
        return

    with open(out_file, "a", encoding="utf-8") as file:
        file.write("\n" + "=" * 70 + "\n\n")
        file.write("Metrics Summary\n\n")

        for metric_data in test_result.metrics_data:
            successful = _is_metric_successful(metric_data)

            if not successful:
                file.write(
                    f"  - ❌ {metric_data.name} (score: {metric_data.score}, threshold: {metric_data.threshold}, "
                    f"strict: {metric_data.strict_mode}, evaluation model: {metric_data.evaluation_model}, "
                    f"reason: {metric_data.reason}, error: {metric_data.error})\n"
                )
            else:
                file.write(
                    f"  - ✅ {metric_data.name} (score: {metric_data.score}, threshold: {metric_data.threshold}, "
                    f"strict: {metric_data.strict_mode}, evaluation model: {metric_data.evaluation_model}, "
                    f"reason: {metric_data.reason}, error: {metric_data.error})\n"
                )

        file.write("\n")
        if test_result.multimodal:
            file.write("For multimodal test case:\n\n")
            file.write(f"  - input: {test_result.input}\n")
            file.write(f"  - actual output: {test_result.actual_output}\n")
        elif test_result.conversational:
            file.write("For conversational test case:\n\n")
            if test_result.turns:
                file.write("  Turns:\n")
                turns = sorted(test_result.turns, key=lambda t: t.order)
                for t in turns:
                    file.write(format_turn(t) + "\n")
            else:
                file.write("  - No turns recorded in this test case.\n")
        else:
            file.write("For test case:\n\n")
            file.write(f"  - input: {test_result.input}\n")
            file.write(f"  - actual output: {test_result.actual_output}\n")
            file.write(f"  - expected output: {test_result.expected_output}\n")
            file.write(f"  - context: {test_result.context}\n")
            file.write(
                f"  - retrieval context: {test_result.retrieval_context}\n"
            )

    aggregate_metric_pass_rates_to_file(
        [test_result] if not isinstance(test_result, list) else test_result
    )


def aggregate_metric_pass_rates(test_results: List[TestResult]) -> dict:
    if not test_results:
        return {}

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


def count_total_metrics_for_trace(trace: Trace) -> int:
    """Span subtree metrics + trace-level metrics."""
    return count_metrics_in_trace(trace=trace) + len(trace.metrics or [])


def count_metrics_in_span_subtree(span: BaseSpan) -> int:
    total = len(span.metrics or [])
    for c in span.children or []:
        total += count_metrics_in_span_subtree(c)
    return total


def extract_trace_test_results(trace_api: TraceApi) -> List[TestResult]:
    test_results: List[TestResult] = []
    # extract trace result
    if trace_api.metrics_data:
        test_results.append(
            TestResult(
                name=trace_api.name,
                success=True,
                metrics_data=trace_api.metrics_data,
                conversational=False,
                input=trace_api.input,
                actual_output=trace_api.output,
                expected_output=trace_api.expected_output,
                context=trace_api.context,
                retrieval_context=trace_api.retrieval_context,
            )
        )
    # extract base span results
    for span in trace_api.base_spans:
        test_results.extend(extract_span_test_results(span))
    # extract agent span results
    for span in trace_api.agent_spans:
        test_results.extend(extract_span_test_results(span))
    # extract llm span results
    for span in trace_api.llm_spans:
        test_results.extend(extract_span_test_results(span))
    # extract retriever span results
    for span in trace_api.retriever_spans:
        test_results.extend(extract_span_test_results(span))
    # extract tool span results
    for span in trace_api.tool_spans:
        test_results.extend(extract_span_test_results(span))

    return test_results


def extract_span_test_results(span_api: BaseApiSpan) -> List[TestResult]:
    test_results: List[TestResult] = []
    if span_api.metrics_data:
        test_results.append(
            TestResult(
                name=span_api.name,
                success=span_api.status == TraceSpanApiStatus.SUCCESS,
                metrics_data=span_api.metrics_data,
                input=span_api.input,
                actual_output=span_api.output,
                expected_output=span_api.expected_output,
                context=span_api.context,
                retrieval_context=span_api.retrieval_context,
                conversational=False,
            )
        )
    return test_results


def count_observe_decorators_in_module(func: Callable) -> int:
    mod = inspect.getmodule(func)
    if mod is None or not hasattr(mod, "__file__"):
        raise RuntimeError("Cannot locate @observe function.")
    module_source = inspect.getsource(mod)
    tree = ast.parse(module_source)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                if (
                    isinstance(deco, ast.Call)
                    and getattr(deco.func, "id", "") == "observe"
                ):
                    count += 1
    return count
