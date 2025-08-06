from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.tracing.context import current_trace_context
from deepeval.tracing.offline_evals.api import EvaluateSpanRequestBody


def evaluate_span(span_uuid: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    api = Api(api_key=api_key)

    evaluate_span_request_body = EvaluateSpanRequestBody(
        spanUuid=span_uuid,
        metricCollection=metric_collection,
    )
    try:
        body = evaluate_span_request_body.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = evaluate_span_request_body.dict(by_alias=True, exclude_none=True)

    api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_SPAN_ENDPOINT,
        body=body,
        url_params={"spanUuid": span_uuid},
    )


async def a_evaluate_span(span_uuid: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    api = Api(api_key=api_key)

    evaluate_span_request_body = EvaluateSpanRequestBody(
        spanUuid=span_uuid,
        metricCollection=metric_collection,
    )
    try:
        body = evaluate_span_request_body.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = evaluate_span_request_body.dict(by_alias=True, exclude_none=True)

    await api.a_send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.EVALUATE_TRACE_ENDPOINT,
        body=body,
        url_params={"spanUuid": span_uuid},
    )
