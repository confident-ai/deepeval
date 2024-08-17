from typing import List, Union

from deepeval.confident.api import (
    Api,
    Endpoints,
    DEEPEVAL_BASE_URL,
    HttpMethods,
)
from deepeval.confident.types import ConfidentEvaluateRequestData
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.utils import is_confident


def run_confident_evaluation(
    experiment_name: str,
    test_cases: List[Union[LLMTestCase, ConversationalTestCase]],
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
            pass
    else:
        raise Exception(
            "To run evaluations on Confident AI, run `deepeval login`."
        )
