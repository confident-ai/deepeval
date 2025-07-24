import json
from opentelemetry.sdk.trace import ReadableSpan
from deepeval.tracing.attributes import LlmAttributes
from deepeval.tracing.types import AgentSpan, BaseSpan, LlmSpan
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

    # conditions to qualify as llm span
    operation_name = span.attributes.get("gen_ai.operation.name")
    request_model = span.attributes.get("gen_ai.request.model")

    if operation_name in ["chat", "generate_content", "text_completion"] and request_model is not None:
        input = []
        output = ""
        try:
            if span.attributes.get("events") is not None:
                events = json.loads(span.attributes.get("events"))
                # Collect all but last assistant message as input, last assistant message as output
                assistant_indices = [i for i, event in enumerate(events) if event.get("message", {}).get("role") == "assistant"]
                if assistant_indices:
                    last_assistant_idx = assistant_indices[-1]
                    # Input: all messages before last assistant message
                    for i, event in enumerate(events[:last_assistant_idx]):
                        msg = event.get("message") or event
                        if msg.get("role") != "assistant":
                            input.append(msg)
                    # Output: content of last assistant message
                    last_assistant_event = events[last_assistant_idx]
                    last_assistant_msg = last_assistant_event.get("message") or last_assistant_event
                    output = last_assistant_msg.get("content", "")
                else:
                    # No assistant message, treat all as input
                    for event in events:
                        msg = event.get("message") or event
                        input.append(msg)
        except Exception as e:
            pass
        model = span.attributes.get("gen_ai.request.model")
        confident_span = LlmSpan(
            uuid=to_hex_string(span.context.span_id, 16),
            status=TraceSpanStatus.SUCCESS, # TODO: handle status
            children=[],
            trace_uuid=active_trace_id,
            start_time=span.start_time/1e9,
            end_time=span.end_time/1e9,
            parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
            name=span.name,
            metadata=json.loads(span.to_json()),
            attributes=LlmAttributes(
                input=input,
                output=output,
            ),
            model=model if model is not None else "unknown",
        )

    return confident_span
    