import os
import json
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from collections import defaultdict
from deepeval.tracing import get_trace_stack
from deepeval.constants import PYTEST_RUN_TEST_NAME, PYTEST_RUN_ENV_VAR


class MetricsMetadata(BaseModel):
    metric: str
    score: float
    minimum_score: float = Field(None, alias="minimumScore")


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
        self.metric_dict = defaultdict(list)
        self.min_score_dict = defaultdict(float)

    def add_metric(self, metric: BaseMetric):
        self.metric_dict[metric.__name__].append(metric.score)
        self.min_score_dict[metric.__name__] = min(
            self.min_score_dict.get(metric.__name__, float("inf")),
            metric.minimum_score,
        )

    def get_metrics_metadata(self):
        return [
            MetricsMetadata(
                metric=metric_name,
                score=sum(scores) / len(scores),
                minimumScore=self.min_score_dict[metric_name],
            )
            for metric_name, scores in self.metric_dict.items()
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
        # TODO: Fix test_file
        "test.py",
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
            context = test_case.context
            success = all([metric.is_successful() for metric in metrics])
            api_test_case: APITestCase = APITestCase(
                name=os.getenv(PYTEST_RUN_TEST_NAME, "-"),
                input=test_case.input,
                actualOutput=test_case.actual_output,
                expectedOutput=test_case.expected_output,
                success=success,
                metricsMetadata=metrics_metadata,
                runDuration=run_duration,
                context=context,
                traceStack=get_trace_stack(),
            )

            self.dict_test_cases[test_case_id] = api_test_case
            self.test_cases.append(api_test_case)

        all_metric_dict = MetricDict()
        for test_case in self.test_cases:
            for metric in test_case.metrics_metadata:
                all_metric_dict.add_metric(metric.metric, metric.score)
        self.metric_scores = all_metric_dict.get_average_metric_score()
        print(self.test_cases)

    def save(self, file_path: Optional[str] = None):
        if file_path is None:
            file_path = os.getenv(PYTEST_RUN_ENV_VAR)
            # If file Path is None, remove it
            if not file_path:
                return
            elif not file_path.endswith(".json"):
                file_path = f"{file_path}.json"
        with open(file_path, "w") as f:
            json.dump(self.dict(by_alias=True, exclude_none=True), f)
        return file_path


class TestRunManger:
    def __init__(self):
        self.test_run = None

    def set_test_run(self, test_run: TestRun):
        self.test_run = test_run

    def get_test_run(self):
        return self.test_run

    def clear_test_run(self):
        self.test_run = None


test_run_manager = TestRunManger()
