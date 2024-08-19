from typing import List, Optional, Union
import webbrowser
from rich.console import Console

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
from deepeval.utils import is_confident


def confident_evaluate(
    experiment_name: str,
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
    disable_browser_opening: Optional[bool] = False,
):
    if is_confident():
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
            body = confident_request_data.dict(by_alias=True, exclude_none=True)

        result = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.EVALUATE_ENDPOINT,
            body=body,
        )

        if result:
            response = ConfidentEvaluateResponseData(
                link=result["link"],
            )
            Console().print(
                f"✅ Evaluation of experiment {experiment_name} starter! View progress on "
                f"[link={response.link}]{response.link}[/link]"
            )

            if disable_browser_opening == False:
                webbrowser.open(response.link)

    else:
        raise Exception(
            "To run evaluations on Confident AI, run `deepeval login`."
        )
