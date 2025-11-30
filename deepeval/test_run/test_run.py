from enum import Enum
import os
import json
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict, Union, Tuple
import shutil
import sys
import datetime
from rich.table import Table
from rich.console import Console
from rich import print


from deepeval.metrics import BaseMetric
from deepeval.confident.api import Api, Endpoints, HttpMethods, is_confident
from deepeval.test_run.api import (
    LLMApiTestCase,
    ConversationalApiTestCase,
    TestRunHttpResponse,
    MetricData,
)
from deepeval.tracing.utils import make_json_serializable
from deepeval.tracing.api import SpanApiType, span_api_type_literals
from deepeval.test_case import LLMTestCase, ConversationalTestCase, MLLMTestCase
from deepeval.utils import (
    delete_file_if_exists,
    get_is_running_deepeval,
    is_read_only_env,
    open_browser,
    shorten,
    format_turn,
    len_short,
)
from deepeval.test_run.cache import global_test_run_cache_manager
from deepeval.constants import CONFIDENT_TEST_CASE_BATCH_SIZE, HIDDEN_DIR
from deepeval.prompt import (
    PromptMessage,
    ModelSettings,
    PromptInterpolationType,
    OutputType,
)
from rich.panel import Panel
from rich.columns import Columns


portalocker = None
if not is_read_only_env():
    try:
        import portalocker
    except Exception as e:
        print(
            f"Warning: failed to import portalocker: {e}",
            file=sys.stderr,
        )
else:
    print(
        "Warning: DeepEval is configured for read only environment. Test runs will not be written to disk."
    )


TEMP_FILE_PATH = f"{HIDDEN_DIR}/.temp_test_run_data.json"
LATEST_TEST_RUN_FILE_PATH = f"{HIDDEN_DIR}/.latest_test_run.json"
LATEST_TEST_RUN_DATA_KEY = "testRunData"
LATEST_TEST_RUN_LINK_KEY = "testRunLink"
console = Console()


class TestRunResultDisplay(Enum):
    ALL = "all"
    FAILING = "failing"
    PASSING = "passing"


class MetricScoreType(BaseModel):
    metric: str
    score: float

    @classmethod
    def from_metric(cls, metric: BaseMetric):
        return cls(metric=metric.__name__, score=metric.score)


class MetricScores(BaseModel):
    metric: str
    scores: List[float]
    passes: int
    fails: int
    errors: int


class TraceMetricScores(BaseModel):
    agent: Dict[str, Dict[str, MetricScores]] = Field(default_factory=dict)
    tool: Dict[str, Dict[str, MetricScores]] = Field(default_factory=dict)
    retriever: Dict[str, Dict[str, MetricScores]] = Field(default_factory=dict)
    llm: Dict[str, Dict[str, MetricScores]] = Field(default_factory=dict)
    base: Dict[str, Dict[str, MetricScores]] = Field(default_factory=dict)


class PromptData(BaseModel):
    alias: Optional[str] = None
    version: Optional[str] = None
    text_template: Optional[str] = None
    messages_template: Optional[List[PromptMessage]] = None
    model_settings: Optional[ModelSettings] = None
    output_type: Optional[OutputType] = None
    interpolation_type: Optional[PromptInterpolationType] = None


class MetricsAverageDict:
    def __init__(self):
        self.metric_dict = {}
        self.metric_count = {}

    def add_metric(self, metric_name, score):
        if metric_name not in self.metric_dict:
            self.metric_dict[metric_name] = score
            self.metric_count[metric_name] = 1
        else:
            self.metric_dict[metric_name] += score
            self.metric_count[metric_name] += 1

    def get_average_metric_score(self):
        return [
            MetricScoreType(
                metric=metric,
                score=self.metric_dict[metric] / self.metric_count[metric],
            )
            for metric in self.metric_dict
        ]


class RemainingTestRun(BaseModel):
    testRunId: str
    test_cases: List[LLMApiTestCase] = Field(
        alias="testCases", default_factory=lambda: []
    )
    conversational_test_cases: List[ConversationalApiTestCase] = Field(
        alias="conversationalTestCases", default_factory=lambda: []
    )


class TestRun(BaseModel):
    test_file: Optional[str] = Field(
        None,
        alias="testFile",
    )
    test_cases: List[LLMApiTestCase] = Field(
        alias="testCases", default_factory=lambda: []
    )
    conversational_test_cases: List[ConversationalApiTestCase] = Field(
        alias="conversationalTestCases", default_factory=lambda: []
    )
    metrics_scores: List[MetricScores] = Field(
        default_factory=lambda: [], alias="metricsScores"
    )
    trace_metrics_scores: Optional[TraceMetricScores] = Field(
        None, alias="traceMetricsScores"
    )
    identifier: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = Field(None)
    prompts: Optional[List[PromptData]] = Field(None)
    test_passed: Optional[int] = Field(None, alias="testPassed")
    test_failed: Optional[int] = Field(None, alias="testFailed")
    run_duration: float = Field(0.0, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    dataset_alias: Optional[str] = Field(None, alias="datasetAlias")
    dataset_id: Optional[str] = Field(None, alias="datasetId")

    def add_test_case(
        self, api_test_case: Union[LLMApiTestCase, ConversationalApiTestCase]
    ):
        if isinstance(api_test_case, ConversationalApiTestCase):
            self.conversational_test_cases.append(api_test_case)
        else:
            self.test_cases.append(api_test_case)

        if api_test_case.evaluation_cost is not None:
            if self.evaluation_cost is None:
                self.evaluation_cost = api_test_case.evaluation_cost
            else:
                self.evaluation_cost += api_test_case.evaluation_cost

    def set_dataset_properties(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    ):
        if self.dataset_alias is None:
            self.dataset_alias = test_case._dataset_alias

        if self.dataset_id is None:
            self.dataset_id = test_case._dataset_id

    def sort_test_cases(self):
        self.test_cases.sort(
            key=lambda x: (x.order if x.order is not None else float("inf"))
        )
        # Optionally update order only if not already set
        highest_order = 0
        for test_case in self.test_cases:
            if test_case.order is None:
                test_case.order = highest_order
            highest_order = test_case.order + 1

        self.conversational_test_cases.sort(
            key=lambda x: (x.order if x.order is not None else float("inf"))
        )
        # Optionally update order only if not already set
        highest_order = 0
        for test_case in self.conversational_test_cases:
            if test_case.order is None:
                test_case.order = highest_order
            highest_order = test_case.order + 1

    def construct_metrics_scores(self) -> int:
        # Use a dict to aggregate scores, passes, and fails for each metric.
        metrics_dict: Dict[str, Dict[str, Any]] = {}
        # Add dict for trace metrics
        trace_metrics_dict: Dict[
            span_api_type_literals, Dict[str, Dict[str, Dict[str, Any]]]
        ] = {
            SpanApiType.AGENT.value: {},
            SpanApiType.TOOL.value: {},
            SpanApiType.RETRIEVER.value: {},
            SpanApiType.LLM.value: {},
            SpanApiType.BASE.value: {},
        }
        valid_scores = 0

        def process_metric_data(metric_data: MetricData):
            """
            Process and aggregate metric data for overall test metrics.

            Args:
                metric_data: The metric data to process
            """
            nonlocal valid_scores
            metric_name = metric_data.name
            score = metric_data.score
            success = metric_data.success

            if metric_name not in metrics_dict:
                metrics_dict[metric_name] = {
                    "scores": [],
                    "passes": 0,
                    "fails": 0,
                    "errors": 0,
                }

            metric_dict = metrics_dict[metric_name]

            if score is None or success is None:
                metric_dict["errors"] += 1
            else:
                valid_scores += 1
                metric_dict["scores"].append(score)
                if success:
                    metric_dict["passes"] += 1
                else:
                    metric_dict["fails"] += 1

        def process_span_metric_data(
            metric_data: MetricData,
            span_type: span_api_type_literals,
            span_name: str,
        ):
            """
            Process and aggregate metric data for a specific span.

            Args:
                metric_data: The metric data to process
                span_type: The type of span (agent, tool, retriever, llm, base)
                span_name: The name of the span
            """
            metric_name = metric_data.name
            score = metric_data.score
            success = metric_data.success

            if span_name not in trace_metrics_dict[span_type]:
                trace_metrics_dict[span_type][span_name] = {}

            if metric_name not in trace_metrics_dict[span_type][span_name]:
                trace_metrics_dict[span_type][span_name][metric_name] = {
                    "scores": [],
                    "passes": 0,
                    "fails": 0,
                    "errors": 0,
                }

            metric_dict = trace_metrics_dict[span_type][span_name][metric_name]

            if score is None or success is None:
                metric_dict["errors"] += 1
            else:
                metric_dict["scores"].append(score)
                if success:
                    metric_dict["passes"] += 1
                else:
                    metric_dict["fails"] += 1

        def process_spans(spans, span_type: span_api_type_literals):
            """
            Process all metrics for a list of spans of a specific type.

            Args:
                spans: List of spans to process
                span_type: The type of spans being processed
            """
            for span in spans:
                if span.metrics_data is not None:
                    for metric_data in span.metrics_data:
                        process_metric_data(metric_data)
                        process_span_metric_data(
                            metric_data, span_type, span.name
                        )

        # Process non-conversational test cases.
        for test_case in self.test_cases:
            if test_case.metrics_data is None:
                continue
            for metric_data in test_case.metrics_data:
                process_metric_data(metric_data)

            if test_case.trace is None:
                continue

            # Process all span types using the helper function
            process_spans(test_case.trace.agent_spans, SpanApiType.AGENT.value)
            process_spans(test_case.trace.tool_spans, SpanApiType.TOOL.value)
            process_spans(
                test_case.trace.retriever_spans, SpanApiType.RETRIEVER.value
            )
            process_spans(test_case.trace.llm_spans, SpanApiType.LLM.value)
            process_spans(test_case.trace.base_spans, SpanApiType.BASE.value)

        # Process conversational test cases.
        for convo_test_case in self.conversational_test_cases:
            if convo_test_case.metrics_data is not None:
                for metric_data in convo_test_case.metrics_data:
                    process_metric_data(metric_data)

        # Create MetricScores objects with the aggregated data.
        self.metrics_scores = [
            MetricScores(
                metric=metric,
                scores=data["scores"],
                passes=data["passes"],
                fails=data["fails"],
                errors=data["errors"],
            )
            for metric, data in metrics_dict.items()
        ]

        # Create a single TraceMetricScores object instead of a list
        trace_metrics_score = TraceMetricScores()
        has_span_metrics = False

        for span_type, spans in trace_metrics_dict.items():
            if not spans:  # Skip empty span types
                continue

            span_dict = {}
            for span_name, metrics in spans.items():
                span_dict[span_name] = {
                    metric_name: MetricScores(
                        metric=metric_name,
                        scores=metric_data["scores"],
                        passes=metric_data["passes"],
                        fails=metric_data["fails"],
                        errors=metric_data["errors"],
                    )
                    for metric_name, metric_data in metrics.items()
                }

            if span_dict:  # Only set if there are spans
                has_span_metrics = True
                setattr(trace_metrics_score, span_type, span_dict)

        # Set to None if no span metrics were found
        self.trace_metrics_scores = (
            trace_metrics_score if has_span_metrics else None
        )
        return valid_scores

    def calculate_test_passes_and_fails(self):
        test_passed = 0
        test_failed = 0
        for test_case in self.test_cases:
            if test_case.success is not None:
                if test_case.success:
                    test_passed += 1
                else:
                    test_failed += 1

        for test_case in self.conversational_test_cases:
            # we don't count for conversational messages success
            if test_case.success is not None:
                if test_case.success:
                    test_passed += 1
                else:
                    test_failed += 1

        self.test_passed = test_passed
        self.test_failed = test_failed

    def save(self, f):
        try:
            body = self.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            body = self.dict(by_alias=True, exclude_none=True)
        json.dump(body, f, cls=TestRunEncoder)
        f.flush()
        os.fsync(f.fileno())
        return self

    @classmethod
    def load(cls, f):
        data: dict = json.load(f)
        return cls(**data)

    def guard_mllm_test_cases(self):
        for test_case in self.test_cases:
            if test_case.is_multimodal():
                raise ValueError(
                    "Unable to send multimodal test cases to Confident AI."
                )


class TestRunEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return make_json_serializable(obj)


class TestRunManager:
    def __init__(self):
        self.test_run = None
        self.temp_file_path = TEMP_FILE_PATH
        self.save_to_disk = False
        self.disable_request = False

    def reset(self):
        self.test_run = None
        self.temp_file_path = TEMP_FILE_PATH
        self.save_to_disk = False
        self.disable_request = False

    def set_test_run(self, test_run: TestRun):
        self.test_run = test_run

    def create_test_run(
        self,
        identifier: Optional[str] = None,
        file_name: Optional[str] = None,
        disable_request: Optional[bool] = False,
    ):
        self.disable_request = disable_request
        test_run = TestRun(
            identifier=identifier,
            testFile=file_name,
            testCases=[],
            metricsScores=[],
            hyperparameters=None,
            testPassed=None,
            testFailed=None,
        )
        self.set_test_run(test_run)

        if self.save_to_disk:
            self.save_test_run(self.temp_file_path)

    def get_test_run(self, identifier: Optional[str] = None):
        if self.test_run is None:
            self.create_test_run(identifier=identifier)

        if portalocker and self.save_to_disk:
            try:
                with portalocker.Lock(
                    self.temp_file_path,
                    mode="r",
                    flags=portalocker.LOCK_SH | portalocker.LOCK_NB,
                ) as file:
                    loaded = self.test_run.load(file)
                    # only overwrite if loading actually worked
                    self.test_run = loaded
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                portalocker.exceptions.LockException,
            ) as e:
                print(
                    f"Warning: Could not load test run from disk: {e}",
                    file=sys.stderr,
                )

        return self.test_run

    def save_test_run(self, path: str, save_under_key: Optional[str] = None):
        if portalocker and self.save_to_disk:
            try:
                # ensure parent directory exists
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

                with portalocker.Lock(path, mode="w") as file:
                    if save_under_key:
                        try:
                            test_run_data = self.test_run.model_dump(
                                by_alias=True, exclude_none=True
                            )
                        except AttributeError:
                            # Pydantic version below 2.0
                            test_run_data = self.test_run.dict(
                                by_alias=True, exclude_none=True
                            )
                        wrapper_data = {save_under_key: test_run_data}
                        json.dump(wrapper_data, file, cls=TestRunEncoder)
                        file.flush()
                        os.fsync(file.fileno())
                    else:
                        self.test_run.save(file)
            except portalocker.exceptions.LockException:
                pass

    def save_final_test_run_link(self, link: str):
        if portalocker:
            try:
                with portalocker.Lock(
                    LATEST_TEST_RUN_FILE_PATH, mode="w"
                ) as file:
                    json.dump({LATEST_TEST_RUN_LINK_KEY: link}, file)
                    file.flush()
                    os.fsync(file.fileno())
            except portalocker.exceptions.LockException:
                pass

    def update_test_run(
        self,
        api_test_case: Union[LLMApiTestCase, ConversationalApiTestCase],
        test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    ):
        if (
            api_test_case.metrics_data is not None
            and len(api_test_case.metrics_data) == 0
            and api_test_case.trace is None
        ):
            return

        if portalocker and self.save_to_disk:
            try:
                with portalocker.Lock(
                    self.temp_file_path,
                    mode="r+",
                    flags=portalocker.LOCK_EX,
                ) as file:
                    file.seek(0)
                    self.test_run = self.test_run.load(file)

                    # Update the test run object
                    self.test_run.add_test_case(api_test_case)
                    self.test_run.set_dataset_properties(test_case)

                    # Save the updated test run back to the file
                    file.seek(0)
                    file.truncate()
                    self.test_run.save(file)
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                portalocker.exceptions.LockException,
            ) as e:
                print(
                    f"Warning: Could not update test run on disk: {e}",
                    file=sys.stderr,
                )
                if self.test_run is None:
                    # guarantee a valid in-memory run so the update can proceed.
                    # never destroy in-memory state on I/O failure.
                    self.create_test_run()
                self.test_run.add_test_case(api_test_case)
                self.test_run.set_dataset_properties(test_case)
        else:
            if self.test_run is None:
                self.create_test_run()

            self.test_run.add_test_case(api_test_case)
            self.test_run.set_dataset_properties(test_case)

    def clear_test_run(self):
        self.test_run = None

    @staticmethod
    def _calculate_success_rate(pass_count: int, fail_count: int) -> str:
        """Calculate success rate percentage or return error message."""
        total = pass_count + fail_count
        if total > 0:
            return str(round((100 * pass_count) / total, 2))
        return "Cannot display metrics for component-level evals, please run 'deepeval view' to see results on Confident AI."

    @staticmethod
    def _get_metric_status(metric_data: MetricData) -> str:
        """Get formatted status string for a metric."""
        if metric_data.error:
            return "[red]ERRORED[/red]"
        elif metric_data.success:
            return "[green]PASSED[/green]"
        return "[red]FAILED[/red]"

    @staticmethod
    def _format_metric_score(metric_data: MetricData) -> str:
        """Format metric score with evaluation details."""
        evaluation_model = metric_data.evaluation_model or "n/a"
        metric_score = (
            round(metric_data.score, 2)
            if metric_data.score is not None
            else None
        )

        return (
            f"{metric_score} "
            f"(threshold={metric_data.threshold}, "
            f"evaluation model={evaluation_model}, "
            f"reason={metric_data.reason}, "
            f"error={metric_data.error})"
        )

    @staticmethod
    def _should_skip_test_case(
        test_case, display: TestRunResultDisplay
    ) -> bool:
        """Determine if test case should be skipped based on display filter."""
        if display == TestRunResultDisplay.PASSING and not test_case.success:
            return True
        elif display == TestRunResultDisplay.FAILING and test_case.success:
            return True
        return False

    @staticmethod
    def _count_metric_results(
        metrics_data: List[MetricData],
    ) -> tuple[int, int]:
        """Count passing and failing metrics."""
        pass_count = 0
        fail_count = 0
        for metric_data in metrics_data:
            if metric_data.success:
                pass_count += 1
            else:
                fail_count += 1
        return pass_count, fail_count

    def _add_test_case_header_row(
        self,
        table: Table,
        test_case_name: str,
        pass_count: int,
        fail_count: int,
    ):
        """Add test case header row with name and success rate."""
        success_rate = self._calculate_success_rate(pass_count, fail_count)
        table.add_row(
            test_case_name,
            *[""] * 3,
            f"{success_rate}%",
        )

    def _add_metric_rows(self, table: Table, metrics_data: List[MetricData]):
        """Add metric detail rows to the table."""
        for metric_data in metrics_data:
            status = self._get_metric_status(metric_data)
            formatted_score = self._format_metric_score(metric_data)

            table.add_row(
                "",
                str(metric_data.name),
                formatted_score,
                status,
                "",
            )

    def _add_separator_row(self, table: Table):
        """Add empty separator row between test cases."""
        table.add_row(*[""] * len(table.columns))

    def display_results_table(
        self, test_run: TestRun, display: TestRunResultDisplay
    ):
        """Display test results in a formatted table."""

        table = Table(title="Test Results")
        column_config = dict(justify="left")
        column_names = [
            "Test case",
            "Metric",
            "Score",
            "Status",
            "Overall Success Rate",
        ]

        for name in column_names:
            table.add_column(name, **column_config)

        # Process regular test cases
        for index, test_case in enumerate(test_run.test_cases):
            if test_case.metrics_data is None or self._should_skip_test_case(
                test_case, display
            ):
                continue
            pass_count, fail_count = self._count_metric_results(
                test_case.metrics_data
            )
            self._add_test_case_header_row(
                table, test_case.name, pass_count, fail_count
            )
            self._add_metric_rows(table, test_case.metrics_data)

            if index < len(test_run.test_cases) - 1:
                self._add_separator_row(table)

        # Process conversational test cases
        for index, conversational_test_case in enumerate(
            test_run.conversational_test_cases
        ):
            if self._should_skip_test_case(conversational_test_case, display):
                continue

            conversational_test_case_name = conversational_test_case.name

            if conversational_test_case.turns:
                turns_table = Table(
                    title=f"Conversation - {conversational_test_case_name}",
                    show_header=True,
                    header_style="bold",
                )
                turns_table.add_column("#", justify="right", width=3)
                turns_table.add_column("Role", justify="left", width=10)

                # subtract fixed widths + borders and padding.
                # ~20 as a safe buffer
                details_max_width = max(
                    48, min(120, console.width - 3 - 10 - 20)
                )
                turns_table.add_column(
                    "Details",
                    justify="left",
                    overflow="fold",
                    max_width=details_max_width,
                )

                # truncate when too long
                tools_max_width = min(60, max(24, console.width // 3))
                turns_table.add_column(
                    "Tools",
                    justify="left",
                    no_wrap=True,
                    overflow="ellipsis",
                    max_width=tools_max_width,
                )

                sorted_turns = sorted(
                    conversational_test_case.turns, key=lambda t: t.order
                )

                for t in sorted_turns:
                    tools = t.tools_called or []
                    tool_names = ", ".join(tc.name for tc in tools)

                    # omit order, role and tools since we show them in a separate columns.
                    details = format_turn(
                        t,
                        include_tools_in_header=False,
                        include_order_role_in_header=False,
                    )

                    turns_table.add_row(
                        str(t.order),
                        t.role,
                        details,
                        shorten(tool_names, len_short()),
                    )

                console.print(turns_table)
            else:
                console.print(
                    f"[dim]No turns recorded for {conversational_test_case_name}.[/dim]"
                )
            if conversational_test_case.metrics_data is not None:
                pass_count, fail_count = self._count_metric_results(
                    conversational_test_case.metrics_data
                )
                self._add_test_case_header_row(
                    table, conversational_test_case.name, pass_count, fail_count
                )
                self._add_metric_rows(
                    table, conversational_test_case.metrics_data
                )

            if index < len(test_run.conversational_test_cases) - 1:
                self._add_separator_row(table)

            if index < len(test_run.test_cases) - 1:
                self._add_separator_row(table)

        table.add_row(
            "[bold red]Note: Use Confident AI with DeepEval to analyze failed test cases for more details[/bold red]",
            *[""] * (len(table.columns) - 1),
        )
        print(table)

    def post_test_run(self, test_run: TestRun) -> Optional[Tuple[str, str]]:
        if (
            len(test_run.test_cases) == 0
            and len(test_run.conversational_test_cases) == 0
        ):
            print("No test cases found, unable to upload to Confident AI.")
            return

        api = Api()

        is_conversational_run = len(test_run.conversational_test_cases) > 0
        all_test_cases_to_process = (
            test_run.conversational_test_cases
            if is_conversational_run
            else test_run.test_cases
        )

        custom_batch_size = os.getenv(CONFIDENT_TEST_CASE_BATCH_SIZE)
        if custom_batch_size and custom_batch_size.isdigit():
            BATCH_SIZE = int(custom_batch_size)
        else:
            BATCH_SIZE = 20 if is_conversational_run else 40

        initial_batch = all_test_cases_to_process[:BATCH_SIZE]
        remaining_test_cases_to_process = all_test_cases_to_process[BATCH_SIZE:]

        if len(remaining_test_cases_to_process) > 0:
            console.print(
                "Sending a large test run to Confident, this might take a bit longer than usual..."
            )

        ####################
        ### POST REQUEST ###
        ####################
        if is_conversational_run:
            test_run.conversational_test_cases = initial_batch
        else:
            test_run.test_cases = initial_batch

        try:
            test_run.prompts = None
            body = test_run.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = test_run.dict(by_alias=True, exclude_none=True)

        json_str = json.dumps(body, cls=TestRunEncoder)
        body = json.loads(json_str)

        data, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.TEST_RUN_ENDPOINT,
            body=body,
        )

        if not isinstance(data, dict) or "id" not in data:
            # try to show helpful details
            detail = None
            if isinstance(data, dict):
                detail = (
                    data.get("detail")
                    or data.get("message")
                    or data.get("error")
                )
            # fall back to repr for visibility
            raise RuntimeError(
                f"Confident API response missing 'id'. "
                f"detail={detail!r} raw={type(data).__name__}:{repr(data)[:500]}"
            )

        res = TestRunHttpResponse(
            id=data["id"],
        )

        ################################################
        ### Send the remaining test cases in batches ###
        ################################################
        total_remaining = len(remaining_test_cases_to_process)
        num_remaining_batches = (
            (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
            if total_remaining > 0
            else 0
        )

        for i in range(num_remaining_batches):
            start_index = i * BATCH_SIZE
            batch = remaining_test_cases_to_process[
                start_index : start_index + BATCH_SIZE
            ]

            if len(batch) == 0:
                break  # Should not happen with correct num_remaining_batches, but as a safeguard

            # Create RemainingTestRun with the correct list populated
            if is_conversational_run:
                remaining_test_run = RemainingTestRun(
                    testRunId=res.id,
                    testCases=[],  # This will be empty
                    conversationalTestCases=batch,
                )
            else:
                remaining_test_run = RemainingTestRun(
                    testRunId=res.id,
                    testCases=batch,
                    conversationalTestCases=[],  # This will be empty
                )

            body = None
            try:
                body = remaining_test_run.model_dump(
                    by_alias=True, exclude_none=True
                )
            except AttributeError:
                # Pydantic version below 2.0
                body = remaining_test_run.dict(by_alias=True, exclude_none=True)

            try:
                _, _ = api.send_request(
                    method=HttpMethods.PUT,
                    endpoint=Endpoints.TEST_RUN_ENDPOINT,
                    body=body,
                )
            except Exception as e:
                message = f"Unexpected error when sending some test cases. Incomplete test run available at {link}"
                raise Exception(message) from e

        console.print(
            "[rgb(5,245,141)]✓[/rgb(5,245,141)] Done 🎉! View results on "
            f"[link={link}]{link}[/link]"
        )
        self.save_final_test_run_link(link)
        open_browser(link)
        return link, res.id

    def save_test_run_locally(self):
        local_folder = os.getenv("DEEPEVAL_RESULTS_FOLDER")
        if local_folder:
            new_test_filename = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            os.rename(self.temp_file_path, new_test_filename)
            if not os.path.exists(local_folder):
                os.mkdir(local_folder)
                shutil.copy(new_test_filename, local_folder)
                print(f"Results saved in {local_folder} as {new_test_filename}")
            elif os.path.isfile(local_folder):
                print(
                    f"""❌ Error: DEEPEVAL_RESULTS_FOLDER={local_folder} already exists and is a file.\nDetailed results won't be saved. Please specify a folder or an available path."""
                )
            else:
                shutil.copy(new_test_filename, local_folder)
                print(f"Results saved in {local_folder} as {new_test_filename}")
            os.remove(new_test_filename)

    def wrap_up_test_run(
        self,
        runDuration: float,
        display_table: bool = True,
        display: Optional[TestRunResultDisplay] = TestRunResultDisplay.ALL,
    ) -> Optional[Tuple[str, str]]:
        test_run = self.get_test_run()
        if test_run is None:
            print("Test Run is empty, please try again.")
            delete_file_if_exists(self.temp_file_path)
            return
        elif (
            len(test_run.test_cases) == 0
            and len(test_run.conversational_test_cases) == 0
        ):
            print("No test cases found, please try again.")
            delete_file_if_exists(self.temp_file_path)
            return

        valid_scores = test_run.construct_metrics_scores()
        if valid_scores == 0:
            print("All metrics errored for all test cases, please try again.")
            delete_file_if_exists(self.temp_file_path)
            delete_file_if_exists(
                global_test_run_cache_manager.temp_cache_file_name
            )
            return
        test_run.run_duration = runDuration
        test_run.calculate_test_passes_and_fails()
        test_run.sort_test_cases()

        if global_test_run_cache_manager.disable_write_cache is None:
            global_test_run_cache_manager.disable_write_cache = not bool(
                get_is_running_deepeval()
            )
        global_test_run_cache_manager.wrap_up_cached_test_run()

        if display_table:
            self.display_results_table(test_run, display)

        if test_run.hyperparameters is None:
            console.print(
                "\n[bold yellow]⚠ WARNING:[/bold yellow] No hyperparameters logged.\n"
                "» [bold blue][link=https://deepeval.com/docs/evaluation-prompts]Log hyperparameters[/link][/bold blue] to attribute prompts and models to your test runs.\n\n"
                + "=" * 80
            )
        else:
            if not test_run.prompts:
                console.print(
                    "\n[bold yellow]⚠ WARNING:[/bold yellow] No prompts logged.\n"
                    "» [bold blue][link=https://deepeval.com/docs/evaluation-prompts]Log prompts[/link][/bold blue] to evaluate and optimize your prompt templates and models.\n\n"
                    + "=" * 80
                )
            else:
                console.print("\n[bold green]✓ Prompts Logged[/bold green]\n")
                self._render_prompts_panels(prompts=test_run.prompts)

        self.save_test_run_locally()
        delete_file_if_exists(self.temp_file_path)
        if is_confident() and self.disable_request is False:
            return self.post_test_run(test_run)
        else:
            self.save_test_run(
                LATEST_TEST_RUN_FILE_PATH,
                save_under_key=LATEST_TEST_RUN_DATA_KEY,
            )
            console.print(
                f"\n\n[rgb(5,245,141)]✓[/rgb(5,245,141)] Evaluation completed 🎉! (time taken: {round(runDuration, 2)}s | token cost: {test_run.evaluation_cost} USD)\n"
                f"» Test Results ({test_run.test_passed + test_run.test_failed} total tests):\n",
                f"  » Pass Rate: {round((test_run.test_passed / (test_run.test_passed + test_run.test_failed)) * 100, 2)}% | Passed: [bold green]{test_run.test_passed}[/bold green] | Failed: [bold red]{test_run.test_failed}[/bold red]\n\n",
                "=" * 80,
                "\n\n» Want to share evals with your team, or a place for your test cases to live? ❤️ 🏡\n"
                "  » Run [bold]'deepeval view'[/bold] to analyze and save testing results on [rgb(106,0,255)]Confident AI[/rgb(106,0,255)].\n\n",
            )

    def get_latest_test_run_data(self) -> Optional[TestRun]:
        try:
            if os.path.exists(LATEST_TEST_RUN_FILE_PATH):
                with open(LATEST_TEST_RUN_FILE_PATH, "r") as file:
                    data = json.load(file)
                    return TestRun.model_validate(
                        data[LATEST_TEST_RUN_DATA_KEY]
                    )
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            pass
        return None

    def get_latest_test_run_link(self) -> Optional[str]:
        try:
            if os.path.exists(LATEST_TEST_RUN_FILE_PATH):
                with open(LATEST_TEST_RUN_FILE_PATH, "r") as file:
                    data = json.load(file)
                    return data[LATEST_TEST_RUN_LINK_KEY]
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            pass
        return None

    def _render_prompts_panels(self, prompts: List[PromptData]) -> None:

        def format_string(
            v, default="[dim]None[/dim]", color: Optional[str] = None
        ):
            formatted_string = str(v) if v not in (None, "", []) else default
            return (
                f"{formatted_string}"
                if color is None or v in (None, "", [])
                else f"[{color}]{formatted_string}[/]"
            )

        panels = []
        for prompt in prompts:
            lines = []
            p_type = (
                "messages"
                if prompt.messages_template
                else ("text" if prompt.text_template else "—")
            )
            if p_type:
                lines.append(f"type: {format_string(p_type, color='blue')}")
            if prompt.output_type:
                lines.append(
                    f"output_type: {format_string(prompt.output_type, color='blue')}"
                )
            if prompt.interpolation_type:
                lines.append(
                    f"interpolation_type: {format_string(prompt.interpolation_type, color='blue')}"
                )
            if prompt.model_settings:
                ms = prompt.model_settings
                settings_lines = [
                    "Model Settings:",
                    f"  – provider: {format_string(ms.provider, color='green')}",
                    f"  – name: {format_string(ms.name, color='green')}",
                    f"  – temperature: {format_string(ms.temperature, color='green')}",
                    f"  – max_tokens: {format_string(ms.max_tokens, color='green')}",
                    f"  – top_p: {format_string(ms.top_p, color='green')}",
                    f"  – frequency_penalty: {format_string(ms.frequency_penalty, color='green')}",
                    f"  – presence_penalty: {format_string(ms.presence_penalty, color='green')}",
                    f"  – stop_sequence: {format_string(ms.stop_sequence, color='green')}",
                    f"  – reasoning_effort: {format_string(ms.reasoning_effort, color='green')}",
                    f"  – verbosity: {format_string(ms.verbosity, color='green')}",
                ]
                lines.append("")
                lines.extend(settings_lines)
            title = f"{format_string(prompt.alias)}"
            if prompt.version:
                title += f" (v{prompt.version})"
            body = "\n".join(lines)
            panel = Panel(
                body,
                title=title,
                title_align="left",
                expand=False,
                padding=(1, 6, 1, 2),
            )
            panels.append(panel)

        if panels:
            console.print(Columns(panels, equal=False, expand=False))


global_test_run_manager = TestRunManager()
