from enum import Enum
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
from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.test_run.api import (
    LLMApiTestCase,
    ConversationalApiTestCase,
    TestRunHttpResponse,
    MetricData,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase, MLLMTestCase
from deepeval.utils import (
    delete_file_if_exists,
    get_is_running_deepeval,
    is_confident,
    is_in_ci_env,
)
from deepeval.test_run.cache import global_test_run_cache_manager
from deepeval.constants import LOGIN_PROMPT

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
    identifier: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = Field(None)
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
            if api_test_case.conversational_instance_id is not None:
                for conversational_test_case in self.conversational_test_cases:
                    if (
                        api_test_case.conversational_instance_id
                        == conversational_test_case.instance_id
                    ):
                        conversational_test_case.turns[api_test_case.order] = (
                            api_test_case
                        )

                        if api_test_case.success is False:
                            conversational_test_case.success = False

                        if conversational_test_case.evaluation_cost is None:
                            conversational_test_case.evaluation_cost = (
                                api_test_case.evaluation_cost
                            )
                        else:
                            if api_test_case.evaluation_cost is not None:
                                conversational_test_case.evaluation_cost += (
                                    api_test_case.evaluation_cost
                                )

                        conversational_test_case.run_duration += (
                            api_test_case.run_duration
                        )
                        break

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

    def delete_test_case_instance_ids(self):
        for conversational_test_case in self.conversational_test_cases:
            del conversational_test_case.instance_id
            for turn in conversational_test_case.turns:
                del turn.conversational_instance_id

        for test_case in self.test_cases:
            if hasattr(test_case, "conversational_instance_id"):
                del test_case.conversational_instance_id

    def construct_metrics_scores(self) -> int:
        # Use a dict to aggregate scores, passes, and fails for each metric.
        metrics_dict: Dict[str, Dict[str, Any]] = {}
        valid_scores = 0

        def process_metric_data(metric_data: MetricData):
            nonlocal valid_scores
            name = metric_data.name
            score = metric_data.score
            success = metric_data.success
            # Initialize dict entry if needed.
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

                # Append the score.
                metrics_dict[name]["scores"].append(score)

                # Increment passes or fails based on the metric_data.success flag.
                if success:
                    metrics_dict[name]["passes"] += 1
                else:
                    metrics_dict[name]["fails"] += 1

        # Process non-conversational test cases.
        for test_case in self.test_cases:
            if test_case.metrics_data is None:
                continue
            for metric_data in test_case.metrics_data:
                process_metric_data(metric_data)

        # Process conversational test cases.
        for convo_test_case in self.conversational_test_cases:
            if convo_test_case.metrics_data is not None:
                for metric_data in convo_test_case.metrics_data:
                    process_metric_data(metric_data)

            for turn in convo_test_case.turns:
                if turn.metrics_data is None:
                    continue
                for metric_data in turn.metrics_data:
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
            # Pydantic version below 2.0
            body = self.dict(by_alias=True, exclude_none=True)
        json.dump(body, f)
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
            self.save_test_run()

    def get_test_run(self, identifier: Optional[str] = None):
        if self.test_run is None:
            self.create_test_run(identifier=identifier)

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
        test_case: Union[LLMTestCase, ConversationalTestCase, MLLMTestCase],
    ):
        if (
            api_test_case.metrics_data is not None
            and len(api_test_case.metrics_data) == 0
        ):
            return

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

    def display_results_table(
        self, test_run: TestRun, display: TestRunResultDisplay
    ):
        table = Table(title="Test Results")
        table.add_column("Test case", justify="left")
        table.add_column("Metric", justify="left")
        table.add_column("Score", justify="left")
        table.add_column("Status", justify="left")
        table.add_column("Overall Success Rate", justify="left")

        for index, test_case in enumerate(test_run.test_cases):
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

            for metric_data in test_case.metrics_data:
                if metric_data.error:
                    status = "[red]ERRORED[/red]"
                elif metric_data.success:
                    status = "[green]PASSED[/green]"
                else:
                    status = "[red]FAILED[/red]"

                evaluation_model = metric_data.evaluation_model
                if evaluation_model is None:
                    evaluation_model = "n/a"

                if metric_data.score is not None:
                    metric_score = round(metric_data.score, 2)
                else:
                    metric_score = None

                table.add_row(
                    "",
                    str(metric_data.name),
                    f"{metric_score} (threshold={metric_data.threshold}, evaluation model={evaluation_model}, reason={metric_data.reason}, error={metric_data.error})",
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

        for index, conversational_test_case in enumerate(
            test_run.conversational_test_cases
        ):
            if (
                display == TestRunResultDisplay.PASSING
                and conversational_test_case.success == False
            ):
                continue
            elif (
                display == TestRunResultDisplay.FAILING
                and conversational_test_case.success
            ):
                continue

            pass_count = 0
            fail_count = 0
            conversational_test_case_name = conversational_test_case.name

            if conversational_test_case.metrics_data is not None:
                for metric_data in conversational_test_case.metrics_data:
                    if metric_data.success:
                        pass_count += 1
                    else:
                        fail_count += 1
                table.add_row(
                    conversational_test_case_name,
                    "",
                    "",
                    "",
                    f"{round((100*pass_count)/(pass_count+fail_count),2)}%",
                )

            if conversational_test_case.metrics_data is not None:
                for metric_data in conversational_test_case.metrics_data:
                    if metric_data.error:
                        status = "[red]ERRORED[/red]"
                    elif metric_data.success:
                        status = "[green]PASSED[/green]"
                    else:
                        status = "[red]FAILED[/red]"

                    evaluation_model = metric_data.evaluation_model
                    if evaluation_model is None:
                        evaluation_model = "n/a"

                    if metric_data.score is not None:
                        metric_score = round(metric_data.score, 2)
                    else:
                        metric_score = None

                    table.add_row(
                        "",
                        str(metric_data.name),
                        f"{metric_score} (threshold={metric_data.threshold}, evaluation model={evaluation_model}, reason={metric_data.reason}, error={metric_data.error})",
                        status,
                        "",
                    )

            for turn in conversational_test_case.turns:
                if turn.metrics_data is None:
                    # skip if no evaluation
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

                for metric_data in test_case.metrics_data:
                    if metric_data.error:
                        status = "[red]ERRORED[/red]"
                    elif metric_data.success:
                        status = "[green]PASSED[/green]"
                    else:
                        status = "[red]FAILED[/red]"

                    evaluation_model = metric_data.evaluation_model
                    if evaluation_model is None:
                        evaluation_model = "n/a"

                    if metric_data.score is not None:
                        metric_score = round(metric_data.score, 2)
                    else:
                        metric_score = None

                    table.add_row(
                        "",
                        str(metric_data.name),
                        f"{metric_score} (threshold={metric_data.threshold}, evaluation model={evaluation_model}, reason={metric_data.reason}, error={metric_data.error})",
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

            if index is not len(self.test_run.test_cases) - 1:
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

    def post_test_run(self, test_run: TestRun) -> Optional[str]:
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
            result = api.send_request(
                method=HttpMethods.POST,
                endpoint=Endpoints.TEST_RUN_ENDPOINT,
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
                    result = api.send_request(
                        method=HttpMethods.PUT,
                        endpoint=Endpoints.TEST_RUN_ENDPOINT,
                        body=body,
                    )
                except Exception as e:
                    message = f"Unexpected error when sending some test cases. Incomplete test run available at {link}"
                    raise Exception(message) from e

            console.print(
                "[rgb(5,245,141)]✓[/rgb(5,245,141)] Tests finished 🎉! View results on "
                f"[link={link}]{link}[/link]."
            )

            if is_in_ci_env() == False:
                webbrowser.open(link)

            return link
        else:
            console.print(
                "\n[rgb(5,245,141)]✓[/rgb(5,245,141)] Tests finished 🎉! Run [bold]'deepeval login'[/bold] to save and analyze evaluation results on Confident AI.\n",
                LOGIN_PROMPT,
                "\n",
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

    def wrap_up_test_run(
        self,
        runDuration: float,
        display_table: bool = True,
        display: Optional[TestRunResultDisplay] = TestRunResultDisplay.ALL,
    ) -> Optional[str]:
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
            delete_file_if_exists(
                global_test_run_cache_manager.temp_cache_file_name
            )
            return
        test_run.run_duration = runDuration
        test_run.calculate_test_passes_and_fails()
        test_run.sort_test_cases()
        test_run.delete_test_case_instance_ids()

        if global_test_run_cache_manager.disable_write_cache is None:
            global_test_run_cache_manager.disable_write_cache = (
                get_is_running_deepeval() == False
            )

        global_test_run_cache_manager.wrap_up_cached_test_run()

        if display_table:
            self.display_results_table(test_run, display)

        self.save_test_run_locally()
        delete_file_if_exists(self.temp_file_name)

        if (
            len(test_run.test_cases) > 0
            or len(test_run.conversational_test_cases) > 0
        ):
            test_run.guard_mllm_test_cases()
            return self.post_test_run(test_run)


global_test_run_manager = TestRunManager()
