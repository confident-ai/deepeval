import os
import json
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from collections import defaultdict
from deepeval.tracing import get_trace_stack
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.decorators.hyperparameters import get_hyperparameters
from deepeval.api import Api, Endpoints
import shutil
import webbrowser
from deepeval.utils import delete_file_if_exists
import sys
import datetime
import portalocker
from rich.table import Table
from rich.console import Console
from rich import print

TEMP_FILE_NAME = "temp_test_run_data.json"


class MetricsMetadata(BaseModel):
    metric: str
    score: float
    minimum_score: float = Field(None, alias="minimumScore")
    reason: Optional[str] = None


class MetricScore(BaseModel):
    metric: str
    score: float

    @classmethod
    def from_metric(cls, metric: BaseMetric):
        return cls(metric=metric.__name__, score=metric.score)


class MetricDict:
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
            MetricScore(
                metric=metric,
                score=self.metric_dict[metric] / self.metric_count[metric],
            )
            for metric in self.metric_dict
        ]


class MetricsMetadataAverageDict:
    def __init__(self):
        self.metric_scores_dict = defaultdict(list)
        self.min_score_dict = defaultdict(float)
        self.metric_reason_dict = defaultdict(str)

    def add_metric(self, metric: BaseMetric):
        metric_name = metric.__name__

        self.metric_scores_dict[metric_name].append(metric.score)
        self.min_score_dict[metric_name] = min(
            self.min_score_dict.get(metric_name, float("inf")),
            metric.minimum_score,
        )
        self.metric_reason_dict[metric_name] = metric.reason

    def get_metrics_metadata(self):
        return [
            MetricsMetadata(
                metric=metric_name,
                score=sum(scores) / len(scores),
                minimumScore=self.min_score_dict[metric_name],
                reason=self.metric_reason_dict[metric_name],
            )
            for metric_name, scores in self.metric_scores_dict.items()
        ]


class APITestCase(BaseModel):
    name: str
    input: str
    actual_output: str = Field(..., alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    success: bool
    metrics_metadata: List[MetricsMetadata] = Field(
        ..., alias="metricsMetadata"
    )
    run_duration: float = Field(..., alias="runDuration")
    traceStack: Optional[dict] = Field(None)
    context: Optional[list] = Field(None)


class TestRun(BaseModel):
    test_file: Optional[str] = Field(
        None,
        alias="testFile",
    )
    dict_test_cases: Dict[int, APITestCase] = Field(
        default_factory=dict,
    )
    test_cases: List[APITestCase] = Field(
        alias="testCases", default_factory=lambda: []
    )

    metric_scores: List[MetricScore] = Field(
        default_factory=lambda: [], alias="metricScores"
    )
    configurations: Optional[dict[Any, Any]] = Field(default_factory=dict)

    def add_llm_test_case(
        self,
        test_case: LLMTestCase,
        metrics: List[BaseMetric],
        run_duration: float,
        index: int,
    ):
        # Check if test case with the same ID already exists
        test_case_id = id(test_case)
        existing_test_case: LLMTestCase = self.dict_test_cases.get(
            test_case_id, None
        )
        metrics_metadata_dict = MetricsMetadataAverageDict()
        for metric in metrics:
            metrics_metadata_dict.add_metric(metric)
        metrics_metadata = metrics_metadata_dict.get_metrics_metadata()

        if existing_test_case:
            # If it exists, append the metrics to the existing test case
            existing_test_case.metrics_metadata.extend(metrics_metadata)
            success = all(
                [
                    metric.score >= metric.minimum_score
                    for metric in existing_test_case.metrics_metadata
                ]
            )
            # Update the success status
            existing_test_case.success = success
        else:
            # If it doesn't exist, create a new test case
            # Adding backwards compatibility to ensure context still works.
            success = all([metric.is_successful() for metric in metrics])
            api_test_case: APITestCase = APITestCase(
                name=os.getenv(PYTEST_RUN_TEST_NAME, f"test_case_{index}"),
                input=test_case.input,
                actualOutput=test_case.actual_output,
                expectedOutput=test_case.expected_output,
                success=success,
                metricsMetadata=metrics_metadata,
                runDuration=run_duration,
                context=test_case.context,
                traceStack=get_trace_stack(),
            )

            self.dict_test_cases[test_case_id] = api_test_case

    def cleanup(self):
        for _, test_case in self.dict_test_cases.items():
            self.test_cases.append(test_case)
        del self.dict_test_cases
        all_metric_dict = MetricDict()
        for test_case in self.test_cases:
            for metric in test_case.metrics_metadata:
                all_metric_dict.add_metric(metric.metric, metric.score)
        self.metric_scores = all_metric_dict.get_average_metric_score()
        self.configurations = get_hyperparameters()

    def save(self, f):
        json.dump(self.dict(by_alias=True, exclude_none=True), f)
        return self

    @classmethod
    def load(cls, f):
        return cls(**json.load(f))


class TestRunHttpResponse(BaseModel):
    testRunId: str
    projectId: str
    link: str


class TestRunManager:
    def __init__(self):
        self.test_run = None
        self.temp_file_name = TEMP_FILE_NAME
        self.save_to_disk = False

    def set_test_run(self, test_run: TestRun):
        self.test_run = test_run

    def create_test_run(self, file_name: Optional[str] = None):
        test_run = TestRun(
            testFile=file_name,
            testCases=[],
            metricScores=[],
            configurations={},
        )
        self.set_test_run(test_run)

        if self.save_to_disk:
            self.save_test_run()

    def get_test_run(self):
        if self.test_run is None or not self.save_to_disk:
            self.create_test_run()

        if self.save_to_disk:
            try:
                with portalocker.Lock(
                    self.temp_file_name, mode="r", timeout=5
                ) as file:
                    self.test_run = self.test_run.load(file)
            except (FileNotFoundError, portalocker.exceptions.LockException):
                print("Error loading test run from disk", file=sys.stderr)
                self.test_run = None
        return self.test_run

    def save_test_run(self):
        if self.save_to_disk:
            try:
                with portalocker.Lock(
                    self.temp_file_name, mode="w", timeout=5
                ) as file:
                    self.test_run = self.test_run.save(file)
            except portalocker.exceptions.LockException:
                print("Error saving test run to disk", file=sys.stderr)

    def clear_test_run(self):
        self.test_run = None

    # TODO: fix table rendering logic
    def display_results_table(self, test_run: TestRun):
        # Calculate the average of each metric
        metrics_avg = {
            metric.metric: metric.score for metric in test_run.metric_scores
        }

        # Count the number of passes and failures
        # Get all the possible metrics first
        all_metrics = {metric.metric for metric in test_run.metric_scores}

        # Loop through to filter for each metric
        passes = {
            metric: len(
                [
                    test_case_metric
                    for test_case in test_run.test_cases
                    for test_case_metric in test_case.metrics_metadata
                    if test_case_metric.metric == metric and test_case.success
                ]
            )
            for metric in all_metrics
        }
        failures = {
            metric: len(
                [
                    test_case_metric
                    for test_case in test_run.test_cases
                    for test_case_metric in test_case.metrics_metadata
                    if test_case_metric.metric == metric
                ]
            )
            - passes[metric]
            for metric in all_metrics
        }

        table = Table(title="Test Results")
        table.add_column("Metric", justify="right")
        table.add_column("Average Score", justify="right")
        table.add_column("Passes", justify="right")
        table.add_column("Failures", justify="right")
        table.add_column("Success Rate", justify="right")
        total_passes = 0
        total_failures = 0
        for metric, avg in metrics_avg.items():
            pass_count = passes[metric]
            fail_count = failures[metric]
            total_passes += pass_count
            total_failures += fail_count
            success_rate = pass_count / (pass_count + fail_count) * 100
            table.add_row(
                metric,
                str(avg),
                f"[green]{str(pass_count)}[/green]",
                f"[red]{str(fail_count)}[/red]",
                f"{success_rate:.2f}%",
            )
        total_tests = total_passes + total_failures
        overall_success_rate = total_passes / total_tests * 100
        table.add_row(
            "Total",
            "-",
            f"[green]{str(total_passes)}[/green]",
            f"[red]{str(total_failures)}[/red]",
            f"{overall_success_rate:.2f}%",
        )
        print(table)

    def post_test_run(self, test_run: TestRun):
        console = Console()

        # TODO: change this, very hacky way to check if api key exists
        if os.path.exists(".deepeval"):
            try:
                body = test_run.model_dump(by_alias=True, exclude_none=True)
            except AttributeError:
                # Pydantic version below 2.0
                body = test_run.dict(by_alias=True, exclude_none=True)
            api = Api()
            result = api.post_request(
                endpoint=Endpoints.CREATE_TEST_RUN_ENDPOINT.value,
                body=body,
            )
            response = TestRunHttpResponse(
                testRunId=result["testRunId"],
                projectId=result["projectId"],
                link=result["link"],
            )
            if response and os.path.exists(".deepeval"):
                link = response.link
                console.print(
                    "✅ Tests finished! View results on "
                    f"[link={link}]{link}[/link]"
                )
                webbrowser.open(link)
        else:
            console.print(
                '✅ Tests finished! Run "deepeval login" to view evaluation results on the web.'
            )

    def save_test_run_locally(self):
        local_folder = os.getenv("DEEPEVAL_RESULTS_FOLDER")
        if local_folder:
            new_test_filename = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            os.rename(self.temp_file_name, new_test_filename)
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

    def wrap_up_test_run(self, display_table: bool = True):
        test_run = test_run_manager.get_test_run()
        test_run.cleanup()
        if test_run is None or len(test_run.test_cases) == 0:
            print("Test Run is empty, please try again.")
            delete_file_if_exists(test_run_manager.temp_file_name)
            return

        if display_table:
            self.display_results_table(test_run)
        self.post_test_run(test_run)
        self.save_test_run_locally()
        delete_file_if_exists(self.temp_file_name)


test_run_manager = TestRunManager()
