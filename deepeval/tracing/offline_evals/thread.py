from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.tracing.context import current_trace_context
from deepeval.utils import is_confident
from deepeval.tracing.offline_evals.api import EvaluateThreadRequestBody


def evaluate_thread(thread_id: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    if not api_key and not is_confident():
        return

    evaluate_thread_request_body = EvaluateThreadRequestBody(
        threadSuppliedId=thread_id,
        metricCollection=metric_collection,
    )
    try:
        body = evaluate_thread_request_body.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = evaluate_thread_request_body.dict(
            by_alias=True, exclude_none=True
        )

    api = Api(api_key=api_key)
    api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_THREAD_ENDPOINT,
        body=body,
        url_params={"threadId": thread_id},
    )


async def a_evaluate_thread(thread_id: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    if not api_key and not is_confident():
        return

    evaluate_thread_request_body = EvaluateThreadRequestBody(
        threadSuppliedId=thread_id,
        metricCollection=metric_collection,
    )
    try:
        body = evaluate_thread_request_body.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = evaluate_thread_request_body.dict(
            by_alias=True, exclude_none=True
        )

    api = Api(api_key=api_key)
    await api.a_send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_THREAD_ENDPOINT,
        body=body,
        url_params={"threadId": thread_id},
    )
