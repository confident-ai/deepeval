import os
import json
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict, Union
import shutil
import webbrowser
import sys
import datetime
import portalocker
from rich.table import Table
from rich.console import Console
from rich import print

from deepeval.metrics import BaseMetric
from deepeval.api import Api, Endpoints
from deepeval.test_run.api import (
    LLMApiTestCase,
    ConversationalApiTestCase,
    TestRunHttpResponse,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.utils import (
    delete_file_if_exists,
    get_is_running_deepeval,
    is_confident,
    is_in_ci_env,
)
from deepeval.test_run.cache import test_run_cache_manager

TEMP_FILE_NAME = "temp_test_run_data.json"


class MetricScoreType(BaseModel):
    metric: str
    score: float

    @classmethod
    def from_metric(cls, metric: BaseMetric):
        return cls(metric=metric.__name__, score=metric.score)


class MetricScores(BaseModel):
    metric: str
    scores: List[float]


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
    hyperparameters: Optional[Dict[Any, Any]] = Field(None)
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
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ):
        if self.dataset_alias is None:
            self.dataset_alias = test_case._dataset_alias

        if self.dataset_id is None:
            self.dataset_id = test_case._dataset_id

    def sort_test_cases(self):
        self.test_cases.sort(
            key=lambda x: (
                x.order if x.order is not None else float("inf"),
                x.name,
            )
        )
        # Optionally update order only if not already set
        highest_order = 0
        for test_case in self.test_cases:
            if test_case.order is None:
                test_case.order = highest_order
            highest_order = test_case.order + 1

        self.conversational_test_cases.sort(
            key=lambda x: (
                x.order if x.order is not None else float("inf"),
                x.name,
            )
        )
        # Optionally update order only if not already set
        highest_order = 0
        for test_case in self.conversational_test_cases:
            if test_case.order is None:
                test_case.order = highest_order
            highest_order = test_case.order + 1

    def construct_metrics_scores(self) -> int:
        metrics_dict: Dict[str, List[float]] = {}
        valid_scores = 0
        for test_case in self.test_cases:
            if test_case.metrics_metadata is None:
                continue
            for metric_metadata in test_case.metrics_metadata:
                metric = metric_metadata.metric
                score = metric_metadata.score
                if score is None:
                    continue
                valid_scores += 1
                if metric in metrics_dict:
                    metrics_dict[metric].append(score)
                else:
                    metrics_dict[metric] = [score]

        for test_case in self.conversational_test_cases:
            # right now, we only look at individual message evaluations
            # and not a conversation as a whole
            for message in test_case.messages:
                if message.metrics_metadata is None:
                    continue
                for metric_metadata in message.metrics_metadata:
                    metric = metric_metadata.metric
                    score = metric_metadata.score
                    if score is None:
                        continue
                    valid_scores += 1
                    if metric in metrics_dict:
                        metrics_dict[metric].append(score)
                    else:
                        metrics_dict[metric] = [score]

        # metrics_scores combines both conversational and nonconvo scores
        # might need to separate in the future
        self.metrics_scores = [
            MetricScores(metric=metric, scores=scores)
            for metric, scores in metrics_dict.items()
        ]
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
            # we don't count for conversational test cases success,
            # because that would be double counting
            for message in test_case.messages:
                if message.success is not None:
                    # check None for messages that are not evaluated
                    if message.success:
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
        json.dump(body, f)
        return self

    @classmethod
    def load(cls, f):
        return cls(**json.load(f))


class TestRunManager:
    def __init__(self):
        self.test_run = None
        self.temp_file_name = TEMP_FILE_NAME
        self.save_to_disk = False
        self.disable_request = False

    def reset(self):
        self.test_run = None
        self.temp_file_name = TEMP_FILE_NAME
        self.save_to_disk = False
        self.disable_request = False

    def set_test_run(self, test_run: TestRun):
        self.test_run = test_run

    def create_test_run(
        self,
        file_name: Optional[str] = None,
        disable_request: Optional[bool] = False,
    ):
        self.disable_request = disable_request
        test_run = TestRun(
            testFile=file_name,
            testCases=[],
            metricsScores=[],
            hyperparameters=None,
            testPassed=None,
            testFailed=None,
        )
        self.set_test_run(test_run)

        if self.save_to_disk:
            self.save_test_run()

    def get_test_run(self):
        if self.test_run is None:
            self.create_test_run()

        if self.save_to_disk:
            try:
                with portalocker.Lock(
                    self.temp_file_name,
                    mode="r",
                    flags=portalocker.LOCK_SH | portalocker.LOCK_NB,
                ) as file:
                    self.test_run = self.test_run.load(file)
            except (
                FileNotFoundError,
                portalocker.exceptions.LockException,
            ) as e:
                print(f"Error loading test run from disk: {e}", file=sys.stderr)
                self.test_run = None

        return self.test_run

    def save_test_run(self):
        if self.save_to_disk:
            try:
                with portalocker.Lock(self.temp_file_name, mode="w") as file:
                    self.test_run = self.test_run.save(file)
            except portalocker.exceptions.LockException:
                print(
                    "In save_test_run, Error saving test run to disk",
                    file=sys.stderr,
                )

    def update_test_run(
        self,
        api_test_case: Union[LLMApiTestCase, ConversationalApiTestCase],
        test_case: Union[LLMTestCase, ConversationalTestCase],
    ):
        if self.save_to_disk:
            try:
                with portalocker.Lock(
                    self.temp_file_name,
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
                portalocker.exceptions.LockException,
            ) as e:
                print(f"Error updating test run to disk: {e}", file=sys.stderr)
                self.test_run = None
        else:
            if self.test_run is None:
                self.create_test_run()

            self.test_run.add_test_case(api_test_case)
            self.test_run.set_dataset_properties(test_case)

    def clear_test_run(self):
        self.test_run = None

    def display_results_table(self, test_run: TestRun):
        table = Table(title="Test Results")
        table.add_column("Test case", justify="left")
        table.add_column("Metric", justify="left")
        table.add_column("Score", justify="left")
        table.add_column("Status", justify="left")
        table.add_column("Overall Success Rate", justify="left")

        for index, test_case in enumerate(test_run.test_cases):
            pass_count = 0
            fail_count = 0
            test_case_name = test_case.name

            for metric_metadata in test_case.metrics_metadata:
                if metric_metadata.success:
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
                if metric_metadata.error:
                    status = "[red]ERRORED[/red]"
                elif metric_metadata.success:
                    status = "[green]PASSED[/green]"
                else:
                    status = "[red]FAILED[/red]"

                evaluation_model = metric_metadata.evaluation_model
                if evaluation_model is None:
                    evaluation_model = "n/a"

                if metric_metadata.score is not None:
                    metric_score = round(metric_metadata.score, 2)
                else:
                    metric_score = None

                table.add_row(
                    "",
                    str(metric_metadata.metric),
                    f"{metric_score} (threshold={metric_metadata.threshold}, evaluation model={evaluation_model}, reason={metric_metadata.reason}, error={metric_metadata.error})",
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

        for index, conversattional_test_case in enumerate(
            test_run.conversational_test_cases
        ):
            for test_case in conversattional_test_case.messages:
                if test_case.metrics_metadata is None:
                    # skip if no evaluation
                    continue

                pass_count = 0
                fail_count = 0
                test_case_name = test_case.name

                for metric_metadata in test_case.metrics_metadata:
                    if metric_metadata.success:
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
                    if metric_metadata.error:
                        status = "[red]ERRORED[/red]"
                    elif metric_metadata.success:
                        status = "[green]PASSED[/green]"
                    else:
                        status = "[red]FAILED[/red]"

                    evaluation_model = metric_metadata.evaluation_model
                    if evaluation_model is None:
                        evaluation_model = "n/a"

                    if metric_metadata.score is not None:
                        metric_score = round(metric_metadata.score, 2)
                    else:
                        metric_score = None

                    table.add_row(
                        "",
                        str(metric_metadata.metric),
                        f"{metric_score} (threshold={metric_metadata.threshold}, evaluation model={evaluation_model}, reason={metric_metadata.reason}, error={metric_metadata.error})",
                        status,
                        "",
                    )

            if index is not len(self.test_run.conversational_test_cases) - 1:
                table.add_row(
                    "",
                    "",
                    "",
                    "",
                    "",
                )

        print(table)
        print(
            f"Total estimated evaluation tokens cost: {test_run.evaluation_cost} USD"
        )

    def post_test_run(self, test_run: TestRun):
        console = Console()

        if is_confident() and self.disable_request is False:
            BATCH_SIZE = 60
            CONVERSATIONAL_BATCH_SIZE = BATCH_SIZE // 3

            initial_batch = test_run.test_cases[:BATCH_SIZE]
            remaining_test_cases = test_run.test_cases[BATCH_SIZE:]

            initial_conversational_batch = test_run.conversational_test_cases[
                :CONVERSATIONAL_BATCH_SIZE
            ]
            remaining_conversational_test_cases = (
                test_run.conversational_test_cases[CONVERSATIONAL_BATCH_SIZE:]
            )

            if (
                len(remaining_test_cases) > 0
                or len(remaining_conversational_test_cases) > 0
            ):
                console.print(
                    "Sending a large test run to Confident, this might take a bit longer than usual..."
                )

            ####################
            ### POST REQUEST ###
            ####################
            test_run.test_cases = initial_batch
            test_run.conversational_test_cases = initial_conversational_batch
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
            link = response.link
            ################################################
            ### Send the remaining test cases in batches ###
            ################################################
            max_iterations = (
                max(
                    len(remaining_test_cases),
                    len(remaining_conversational_test_cases),
                )
                // CONVERSATIONAL_BATCH_SIZE
            )
            for i in range(0, max_iterations + 1):
                test_case_index = (
                    i * CONVERSATIONAL_BATCH_SIZE * 3
                )  # Multiply by 3 to match the conversational batch step
                test_case_batch = remaining_test_cases[
                    test_case_index : test_case_index + BATCH_SIZE
                ]

                # Adjusting for conversational_test_cases
                conversational_index = i * CONVERSATIONAL_BATCH_SIZE
                conversational_batch = remaining_conversational_test_cases[
                    conversational_index : conversational_index
                    + CONVERSATIONAL_BATCH_SIZE
                ]

                if len(test_case_batch) == 0 and len(conversational_batch) == 0:
                    break

                remaining_test_run = RemainingTestRun(
                    testRunId=response.testRunId,
                    testCases=test_case_batch,
                    conversationalTestCases=conversational_batch,
                )

                body = None
                try:
                    body = remaining_test_run.model_dump(
                        by_alias=True, exclude_none=True
                    )
                except AttributeError:
                    # Pydantic version below 2.0
                    body = remaining_test_run.dict(
                        by_alias=True, exclude_none=True
                    )

                try:
                    result = api.put_request(
                        endpoint=Endpoints.TEST_RUN_ENDPOINT.value,
                        body=body,
                    )
                except Exception as e:
                    message = f"Unexpected error when sending some test cases. Incomplete test run available at {link}"
                    raise Exception(message) from e

            console.print(
                "✅ Tests finished! View results on "
                f"[link={link}]{link}[/link]"
            )

            if is_in_ci_env() == False:
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

    def wrap_up_test_run(self, runDuration: float, display_table: bool = True):
        test_run = self.get_test_run()
        if test_run is None:
            print("Test Run is empty, please try again.")
            delete_file_if_exists(self.temp_file_name)
            return
        elif (
            len(test_run.test_cases) == 0
            and len(test_run.conversational_test_cases) == 0
        ):
            print("No test cases found, please try again.")
            delete_file_if_exists(self.temp_file_name)
            return

        valid_scores = test_run.construct_metrics_scores()
        if valid_scores == 0:
            print("All metrics errored for all test cases, please try again.")
            delete_file_if_exists(self.temp_file_name)
            delete_file_if_exists(test_run_cache_manager.temp_cache_file_name)
            return
        test_run.run_duration = runDuration
        test_run.calculate_test_passes_and_fails()
        test_run.sort_test_cases()

        if test_run_cache_manager.disable_write_cache is None:
            test_run_cache_manager.disable_write_cache = (
                get_is_running_deepeval() == False
            )

        test_run_cache_manager.wrap_up_cached_test_run()

        if display_table:
            self.display_results_table(test_run)

        self.save_test_run_locally()
        delete_file_if_exists(self.temp_file_name)
        self.post_test_run(test_run)


test_run_manager = TestRunManager()
