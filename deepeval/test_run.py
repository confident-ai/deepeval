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


class MetricScoreType(BaseModel):
    metric: str
    score: float

    @classmethod
    def from_metric(cls, metric: BaseMetric):
        return cls(metric=metric.__name__, score=metric.score)


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
    id: Optional[str] = None


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

    metric_scores: List[MetricScoreType] = Field(
        default_factory=lambda: [], alias="metricScores"
    )
    configurations: Optional[dict[Any, Any]] = Field(default_factory=dict)

    def add_llm_test_case(
        self,
        test_case: LLMTestCase,
        metric: BaseMetric,
        run_duration: float,
        index: int,
    ):
        # Check if test case with the same ID already exists
        test_case_id = id(test_case)
        existing_test_case: LLMTestCase = self.dict_test_cases.get(
            test_case_id, None
        )

        metrics_metadata = MetricsMetadata(
            metric=metric.__name__,
            score=metric.score,
            minimumScore=0.5,
            reason=metric.reason,
        )

        if existing_test_case:
            # If it exists, append the metrics to the existing test case
            existing_test_case.metrics_metadata.append(metrics_metadata)
            success = all(
                [
                    metric.score >= metric.minimum_score
                    for metric in existing_test_case.metrics_metadata
                ]
            )
            # Update the success status
            existing_test_case.success = success
        else:
            api_test_case: APITestCase = APITestCase(
                name=os.getenv(PYTEST_RUN_TEST_NAME, f"test_case_{index}"),
                input=test_case.input,
                actualOutput=test_case.actual_output,
                expectedOutput=test_case.expected_output,
                success=metric.is_successful(),
                metricsMetadata=[metrics_metadata],
                runDuration=run_duration,
                context=test_case.context,
                traceStack=get_trace_stack(),
                id=test_case.id,
            )
            self.dict_test_cases[test_case_id] = api_test_case

    def cleanup(self):
        for _, test_case in self.dict_test_cases.items():
            self.test_cases.append(test_case)
        del self.dict_test_cases
        all_metric_dict = MetricsAverageDict()
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

    def display_results_table(self, test_run: TestRun):
        table = Table(title="Test Results")
        table.add_column("Test case", justify="right")
        table.add_column("Metric", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="right")
        table.add_column("Overall Success Rate", justify="right")

        for index, test_case in enumerate(test_run.test_cases):
            pass_count = 0
            fail_count = 0
            test_case_name = test_case.name
            if test_case.id:
                test_case_name += f" ({test_case.id})"

            for metric_metadata in test_case.metrics_metadata:
                if metric_metadata.score >= metric_metadata.score:
                    pass_count += 1
                else:
                    fail_count += 1

            table.add_row(
                test_case_name,
                "",
                "",
                "",
                f"{round((100*pass_count)/(pass_count+fail_count),2)}%",
            )

            for metric_metadata in test_case.metrics_metadata:
                if metric_metadata.score >= metric_metadata.minimum_score:
                    status = "[green]PASSED[/green]"
                else:
                    status = "[red]FAILED[/red]"

                table.add_row(
                    "",
                    str(metric_metadata.metric),
                    f"{round(metric_metadata.score,2)} (threshold={metric_metadata.minimum_score})",
                    status,
                    "",
                )

            if index is not len(self.test_run.test_cases) - 1:
                table.add_row(
                    "",
                    "",
                    "",
                    "",
                    "",
                )
        print(table)

    def post_test_run(self, test_run: TestRun):
        console = Console()

        for test_case in test_run.test_cases:
            test_case.id = None

        if os.path.exists(".deepeval"):
            try:
                body = test_run.model_dump(by_alias=True, exclude_none=True)
            except AttributeError:
                # Pydantic version below 2.0
                body = test_run.dict(by_alias=True, exclude_none=True)
            api = Api()
            result = api.post_request(
                endpoint=Endpoints.TEST_RUN_ENDPOINT.value,
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
        if test_run is None:
            print("Test Run is empty, please try again.")
            delete_file_if_exists(test_run_manager.temp_file_name)
            return
        elif len(test_run.test_cases) == 0:
            print("No test cases found, please try again.")
            delete_file_if_exists(test_run_manager.temp_file_name)
            return

        if display_table:
            self.display_results_table(test_run)
        self.post_test_run(test_run)
        self.save_test_run_locally()
        delete_file_if_exists(self.temp_file_name)


test_run_manager = TestRunManager()
