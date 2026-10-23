import json
from time import perf_counter

from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import BaseSpan, Trace, TraceSpanStatus


class ObjectResult:
    def __init__(self):
        self.message = "ok"


def test_trace_api_payload_serializes_object_inputs_outputs_and_metadata():
    span = BaseSpan(
        uuid="span-1",
        status=TraceSpanStatus.SUCCESS,
        trace_uuid="trace-1",
        start_time=perf_counter(),
        end_time=perf_counter(),
        name="object span",
        input={"request": ObjectResult()},
        output=ObjectResult(),
        metadata={"raw": ObjectResult()},
    )
    trace = Trace(
        uuid="trace-1",
        status=TraceSpanStatus.SUCCESS,
        root_spans=[span],
        start_time=perf_counter(),
        end_time=perf_counter(),
        input=ObjectResult(),
        output={"response": ObjectResult()},
        metadata={"trace": ObjectResult()},
    )

    body = trace_manager.create_trace_api(trace).model_dump(
        by_alias=True, exclude_none=True
    )

    json.dumps(body)
    assert body["input"] == {"message": "ok"}
    assert body["output"] == {"response": {"message": "ok"}}
    assert body["metadata"] == {"trace": {"message": "ok"}}
    assert body["baseSpans"][0]["input"] == {"request": {"message": "ok"}}
    assert body["baseSpans"][0]["output"] == {"message": "ok"}
    assert body["baseSpans"][0]["metadata"] == {"raw": {"message": "ok"}}
