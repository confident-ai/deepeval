from typing import Any, Optional, List, Dict, Union
from pydantic import BaseModel, Field
from rich.table import Table
from rich import print
from enum import Enum
import json

from deepeval.tracing.api import AgenticApiTestCase
from deepeval.test_run.api import MetricData
from deepeval.metrics import BaseMetric
from deepeval.dataset import Golden


TEMP_FILE_NAME = "temp_test_run_data.json"


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


class RemainingTestRun(BaseModel):
    testRunId: str
    agentic_test_cases: List[AgenticApiTestCase] = Field(
        alias="agenticTestCases", default_factory=lambda: []
    )

#############################################
## Agentic Test Run
#############################################

class AgenticTestRun(BaseModel):
    agentic_test_cases: List[AgenticApiTestCase] = Field(
        alias="agenticTestCases", default_factory=lambda: []
    )
    metrics_scores: List[MetricScores] = Field(
        default_factory=lambda: [], alias="metricsScores"
    )
    identifier: Optional[str] = None
    test_passed: Optional[int] = Field(None, alias="testPassed")
    test_failed: Optional[int] = Field(None, alias="testFailed")
    run_duration: float = Field(0.0, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    dataset_alias: Optional[str] = Field(None, alias="datasetAlias")
    dataset_id: Optional[str] = Field(None, alias="datasetId")

    def add_test_case(
        self, api_agentic_test_case: AgenticApiTestCase
    ):
        self.agentic_test_cases.append(api_agentic_test_case)
        if api_agentic_test_case.evaluation_cost is not None:
            if self.evaluation_cost is None:
                self.evaluation_cost = api_agentic_test_case.evaluation_cost
            else:
                self.evaluation_cost += api_agentic_test_case.evaluation_cost

    def set_dataset_properties(
        self,
        golden: Golden,
    ):
        if self.dataset_alias is None:
            self.dataset_alias = golden.dataset_alias
        if self.dataset_id is None:
            self.dataset_id = golden.dataset_id

    def sort_test_cases(self):
        self.agentic_test_cases.sort(
            key=lambda x: (x.order if x.order is not None else float("inf"))
        )
        # Optionally update order only if not already set
        highest_order = 0
        for api_agentic_test_case in self.agentic_test_cases:
            if api_agentic_test_case.order is None:
                api_agentic_test_case.order = highest_order
            highest_order = api_agentic_test_case.order + 1

    def construct_metrics_scores(self) -> int:
        # Use a dict to aggregate scores, passes, and fails for each metric.
        metrics_dict: Dict[str, Dict[str, Any]] = {}
        valid_scores = 0

        def process_metric_data(metric_data: MetricData):
            nonlocal valid_scores
            name = metric_data.name
            score = metric_data.score
            success = metric_data.success
            if name not in metrics_dict:
                metrics_dict[name] = {
                    "scores": [],
                    "passes": 0,
                    "fails": 0,
                    "errors": 0,
                }
            if score is None or success is None:
                metrics_dict[name]["errors"] += 1
            else:
                valid_scores += 1
                metrics_dict[name]["scores"].append(score)
                if success:
                    metrics_dict[name]["passes"] += 1
                else:
                    metrics_dict[name]["fails"] += 1

        # Process non-conversational test cases.
        for agentic_test_case in self.agentic_test_cases:
            if agentic_test_case.metrics_data is None:
                continue
            for metric_data in agentic_test_case.metrics_data:
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
        return valid_scores

    def calculate_test_passes_and_fails(self):
        test_passed = 0
        test_failed = 0
        for test_case in self.agentic_test_cases:
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
            # Pydantic version below 2.0
            body = self.dict(by_alias=True, exclude_none=True)
        json.dump(body, f, cls=TestRunEncoder)
        return self

    @classmethod
    def load(cls, f):
        data: dict = json.load(f)
        return cls(**data)


class TestRunEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


#############################################
## Agentic Test Run Manager
#############################################

class AgenticTestRunManager:
    def __init__(self):
        self.test_run = None
        self.disable_request = False

    def reset(self):
        self.test_run = None
        self.disable_request = False

    def set_test_run(self, test_run: AgenticTestRun):
        self.test_run = test_run

    def create_test_run(
        self,
        identifier: Optional[str] = None,
        file_name: Optional[str] = None,
        disable_request: Optional[bool] = False,
    ):
        self.disable_request = disable_request
        test_run = AgenticTestRun(
            identifier=identifier,
            testFile=file_name,
            agentic_test_cases=[],
            metricsScores=[],
            testPassed=None,
            testFailed=None,
        )
        self.set_test_run(test_run)

    def get_test_run(self, identifier: Optional[str] = None):
        if self.test_run is None:
            self.create_test_run(identifier=identifier)
        return self.test_run

    def update_test_run(
        self,
        api_test_case: AgenticApiTestCase,
        golden: Golden,
    ):
        if (
            api_test_case.metrics_data is not None
            and len(api_test_case.metrics_data) == 0
        ):
            return
        if self.test_run is None:
            self.create_test_run()
        self.test_run.add_test_case(api_test_case)
        self.test_run.set_dataset_properties(golden)

    def clear_test_run(self):
        self.test_run = None

    def post_test_run(self, test_run: AgenticTestRun) -> Optional[str]:
        # print(test_run.agentic_test_cases)
        pass

    def wrap_up_test_run(
        self,
        runDuration: float,
        display_table: bool = True,
        display: Optional[TestRunResultDisplay] = TestRunResultDisplay.ALL,
    ) -> Optional[str]:
        test_run = self.get_test_run()
        if test_run is None:
            print("Test Run is empty, please try again.")
            return
        valid_scores = test_run.construct_metrics_scores()
        if valid_scores == 0:
            print("All metrics errored for all test cases, please try again.")
            return
        test_run.run_duration = runDuration
        test_run.calculate_test_passes_and_fails()
        test_run.sort_test_cases()
        if display_table:
            self.display_results_table(test_run, display)

        if (
            len(test_run.agentic_test_cases) > 0
        ):
            return self.post_test_run(test_run)

    def display_results_table(
        self, test_run: AgenticTestRun, display: TestRunResultDisplay
    ):
        table = Table(title="Test Results")
        table.add_column("Test case", justify="left")
        table.add_column("Metric", justify="left")
        table.add_column("Average Score", justify="left")
        table.add_column("Status", justify="left")
        table.add_column("Overall Success Rate", justify="left")

        for index, test_case in enumerate(test_run.agentic_test_cases):
            if test_case.metrics_data is None:
                continue

            if (
                display == TestRunResultDisplay.PASSING
                and test_case.success == False
            ):
                continue
            elif display == TestRunResultDisplay.FAILING and test_case.success:
                continue

            pass_count = 0
            fail_count = 0
            test_case_name = test_case.name

            for metric_data in test_case.metrics_data:
                if metric_data.success:
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

            grouped: Dict[str, List[MetricData]] = {}
            for md in test_case.metrics_data:
                grouped.setdefault(md.name, []).append(md)

            for metric_name, entries in grouped.items():
                # 1) average score (skip None)
                scores = [e.score for e in entries if e.score is not None]
                avg_score = round(sum(scores) / len(scores), 2) if scores else None

                # 2) build per-run details, and track any errored/failed flags
                details = []
                any_errored = False
                any_failed = False

                for e in entries:
                    # per-run status
                    if e.error:
                        status_str = "[red]ERRORED[/red]"
                        any_errored = True
                    elif e.success:
                        status_str = "[green]PASSED[/green]"
                    else:
                        status_str = "[red]FAILED[/red]"
                        any_failed = True

                    eval_model = e.evaluation_model or "n/a"
                    metric_score = round(e.score, 2) if e.score is not None else None

                    details.append(
                        f"(score={metric_score}, threshold={e.threshold}, "
                        f"evaluation model={eval_model}, reason={e.reason}, "
                        f"error={e.error}, status={status_str})"
                    )

                # 3) overall status
                if any_errored:
                    overall_status = "[red]ERRORED[/red]"
                elif any_failed:
                    overall_status = "[red]FAILED[/red]"
                else:
                    overall_status = "[green]PASSED[/green]"

                # 4) add a single row for this metric
                table.add_row(
                    "",
                    str(metric_name),
                    f"{avg_score} (Average), Breakdown: {details}",
                    overall_status,
                    "",
                )

            if index is not len(self.test_run.agentic_test_cases) - 1:
                table.add_row(
                    "",
                    "",
                    "",
                    "",
                    "",
                )

        table.add_row(
            "[bold red]Note: Use Confident AI with DeepEval to analyze failed test cases for more details[/bold red]",
            "",
            "",
            "",
            "",
        )
        print(table)
        print(
            f"Total estimated evaluation tokens cost: {test_run.evaluation_cost} USD"
        )

global_agentic_test_run_manager = AgenticTestRunManager()
