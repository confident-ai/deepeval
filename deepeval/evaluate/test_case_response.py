from typing import Any, Dict, List, Optional

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.evaluate.api import SendTestCaseResponseApiBody
from deepeval.test_case import ToolCall


def send_test_case_response(
    test_case_id: str,
    actual_output: Optional[str] = None,
    retrieval_context: Optional[List[str]] = None,
    tools_called: Optional[List[ToolCall]] = None,
    expected_tools: Optional[List[ToolCall]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Any:
    api = Api(api_key=api_key)

    request_body = SendTestCaseResponseApiBody(
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        tools_called=tools_called,
        expected_tools=expected_tools,
        metadata=metadata,
    )
    body = request_body.model_dump(by_alias=True, exclude_none=True, mode="json")

    data, _ = api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.TEST_RUN_EVALUATE_ENDPOINT,
        body=body,
        url_params={"testCaseId": test_case_id},
    )
    return data
