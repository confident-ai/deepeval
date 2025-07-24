import json
from opentelemetry.sdk.trace import ReadableSpan
from deepeval.tracing.types import AgentSpan, BaseSpan
from deepeval.tracing.otel.utils import to_hex_string
from deepeval.tracing.types import TraceSpanStatus

def otel_span_handler(span: ReadableSpan, active_trace_id: str) -> BaseSpan:
    
    # default span
    confident_span = BaseSpan(
        uuid=to_hex_string(span.context.span_id, 16),
        status=TraceSpanStatus.SUCCESS, # TODO: handle status
        children=[],
        trace_uuid=active_trace_id,
        start_time=span.start_time/1e9,
        end_time=span.end_time/1e9,
        parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
        name=span.name,
        metadata=json.loads(span.to_json())
    )

    # conditions to qualify as agent span
    if span.attributes.get("agent_name") is not None:
        input = ""
        if span.attributes.get("all_messages_events") is not None:
            input = span.attributes.get("all_messages_events")
        
        output = ""
        if span.attributes.get("final_result") is not None:
            output = span.attributes.get("final_result")
        
        confident_span = AgentSpan(
            uuid=to_hex_string(span.context.span_id, 16),
            status=TraceSpanStatus.SUCCESS, # TODO: handle status
            children=[],
            trace_uuid=active_trace_id,
            start_time=span.start_time/1e9,
            end_time=span.end_time/1e9,
            parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
            name=span.name,
            metadata=json.loads(span.to_json()),
            input = input,
            output = output,
        )
    return confident_span
    