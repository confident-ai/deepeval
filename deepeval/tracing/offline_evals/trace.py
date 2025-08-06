from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.offline_evals.api import EvaluateTraceRequestBody


def evaluate_trace(trace_uuid: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    api = Api(api_key=api_key)

    evaluate_trace_request_body = EvaluateTraceRequestBody(
        traceUuid=trace_uuid,
        metricCollection=metric_collection,
    )
    try:
        body = evaluate_trace_request_body.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = evaluate_trace_request_body.dict(
            by_alias=True, exclude_none=True
        )

    api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_TRACE_ENDPOINT,
        body=body,
        url_params={"traceUuid": trace_uuid},
    )


async def a_evaluate_trace(trace_uuid: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    api = Api(api_key=api_key)

    evaluate_trace_request_body = EvaluateTraceRequestBody(
        traceUuid=trace_uuid,
        metricCollection=metric_collection,
    )
    try:
        body = evaluate_trace_request_body.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = evaluate_trace_request_body.dict(
            by_alias=True, exclude_none=True
        )

    await api.a_send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_TRACE_ENDPOINT,
        body=body,
        url_params={"traceUuid": trace_uuid},
    )
