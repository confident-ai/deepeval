import time
from typing import List, Optional, Union
import webbrowser
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from deepeval.confident.api import (
    Api,
    Endpoints,
    DEEPEVAL_BASE_URL,
    HttpMethods,
)
from deepeval.confident.types import (
    ConfidentEvaluateRequestData,
    ConfidentEvaluateResponseData,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.test_case.utils import check_valid_test_cases_type
from deepeval.utils import is_confident


def confident_evaluate(
    experiment_name: str,
    test_cases: Union[List[LLMTestCase], List[ConversationalTestCase]],
    disable_browser_opening: Optional[bool] = False,
):
    check_valid_test_cases_type(test_cases)

    if is_confident():
        response = None
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"Sending {len(test_cases)} test case(s) to [rgb(106,0,255)]Confident AI[/rgb(106,0,255)]...",
                total=100,
            )
            start_time = time.perf_counter()

            api = Api(base_url=DEEPEVAL_BASE_URL)
            confident_request_data = ConfidentEvaluateRequestData(
                experimentName=experiment_name, testCases=test_cases
            )
            try:
                body = confident_request_data.model_dump(
                    by_alias=True, exclude_none=True
                )
            except AttributeError:
                # Pydantic version below 2.0
                body = confident_request_data.dict(
                    by_alias=True, exclude_none=True
                )

            try:
                result = api.send_request(
                    method=HttpMethods.POST,
                    endpoint=Endpoints.EVALUATE_ENDPOINT,
                    body=body,
                )
            except Exception as e:
                end_time = time.perf_counter()
                time_taken = format(end_time - start_time, ".2f")
                finished_description = f"{progress.tasks[task_id].description} [rgb(245,5,57)]Errored! ({time_taken}s)"
                progress.update(task_id, description=finished_description)
                raise e

            end_time = time.perf_counter()
            time_taken = format(end_time - start_time, ".2f")
            if result:
                response = ConfidentEvaluateResponseData(
                    link=result["link"],
                )

                finished_description = f"{progress.tasks[task_id].description} [rgb(25,227,160)]Done! ({time_taken}s)"
                progress.update(
                    task_id,
                    description=finished_description,
                )

        if response:
            Console().print(
                f"[rgb(5,245,141)]✓[/rgb(5,245,141)] Evaluation of experiment '{experiment_name}' started! View progress on "
                f"[link={response.link}]{response.link}[/link]"
            )

            if disable_browser_opening == False:
                webbrowser.open(response.link)

    else:
        raise Exception(
            "To run evaluations on Confident AI, run `deepeval login`."
        )
