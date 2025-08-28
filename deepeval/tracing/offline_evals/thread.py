from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.offline_evals.api import EvaluateThreadRequestBody


def evaluate_thread(
    thread_id: str, metric_collection: str, overwrite_metrics: bool = False
):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    api = Api(api_key=api_key)

    evaluate_thread_request_body = EvaluateThreadRequestBody(
        metricCollection=metric_collection,
        overwriteMetrics=overwrite_metrics,
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

    api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_THREAD_ENDPOINT,
        body=body,
        url_params={"threadId": thread_id},
    )


async def a_evaluate_thread(
    thread_id: str, metric_collection: str, overwrite_metrics: bool = False
):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    api = Api(api_key=api_key)

    evaluate_thread_request_body = EvaluateThreadRequestBody(
        metricCollection=metric_collection,
        overwriteMetrics=overwrite_metrics,
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

    await api.a_send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_THREAD_ENDPOINT,
        body=body,
        url_params={"threadId": thread_id},
    )
