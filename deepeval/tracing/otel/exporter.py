from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from deepeval.tracing.otel.types import (
    BaseConfidentGenAiOperationSpan,
    ConfidentLlmInputMessage,
    ConfidentLlmOutput,
    ConfidentLlmSpan
)
from typing import List, Optional, Sequence
from datetime import datetime, timezone
from deepeval.tracing.api import TraceSpanApiStatus
from deepeval.tracing.otel.utils import to_hex_string
from deepeval.tracing.tracing import BaseApiSpan, SpanApiType, TraceApi, to_zod_compatible_iso
import typing


class DeepEvalSpanExporter(SpanExporter):
    def __init__(self):
        super().__init__()

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        # convert spans to confident spans
        confident_spans = [self.convert_to_confident_span(span) for span in spans]
        # convert confident spans to base api spans
        base_api_spans = self.convert_confident_genai_operation_spans_to_base_api_spans(confident_spans)
        # aggregate base api spans to traces
        traces = self.aggregate_base_api_spans_to_traces(base_api_spans)
        # return traces
        self._post_traces(traces)

        return SpanExportResult.SUCCESS

    def shutdown(self):
        # Clean up resources, if any
        pass

    def force_flush(self, timeout_millis=30000):
        # Flush any buffered data, if needed
        return True
    
    def _post_traces(self, traces: List[TraceApi]):
        for trace in traces:
            body = trace.model_dump(
                by_alias=True,
                exclude_none=True,
            )
            print(body)
            print("--------------------------------")

    def convert_to_confident_span(self, span: ReadableSpan) -> Optional[BaseConfidentGenAiOperationSpan]:
        """
        Convert a ReadableSpan to a ConfidentGenAiOperationSpan.
        
        Args:
            span: A ReadableSpan object from OpenTelemetry
            
        Returns:
            A ConfidentGenAiOperationSpan object or None if the span is not relevant to Gen AI operations
        """
        # Extract standard span information
        trace_id = to_hex_string(span.context.trace_id)
        span_id = to_hex_string(span.context.span_id)
        parent_span_id = to_hex_string(span.parent.span_id) if span.parent else None
        
        # Skip if we don't have a valid trace_id or span_id
        if not trace_id or not span_id:
            return None
        
        # Extract start and end times
        start_time_unix_nano = span.start_time
        end_time_unix_nano = span.end_time
        
        # Extract status information
        status_code = "OK" if span.status.status_code == StatusCode.OK else "ERROR"
        status_message = span.status.description
        
        # Extract Gen AI specific attributes from span attributes
        attributes = span.attributes or {}

        # Decide the span type from operation name
        # ref: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
        operation_name = attributes.get("gen_ai.operation.name")
        span_type = SpanApiType.BASE.value
        # For LLM Spans
        if operation_name in ["chat", "generate_content", "text_completion"]:
            span_type = SpanApiType.LLM.value
            # Extract model information
            model = attributes.get("gen_ai.request.model")
            
            # Extract token usage information
            input_tokens = attributes.get("gen_ai.usage.input_tokens")
            output_tokens = attributes.get("gen_ai.usage.output_tokens")
            
            # Process input messages
            input_messages = []
            system_message = attributes.get("gen_ai.system.message")
            user_message = attributes.get("gen_ai.user.message")
            assistant_input_message = attributes.get("gen_ai.assistant.message") 
            
                
            # Create input messages if any exist
            if system_message:
                input_messages.append(ConfidentLlmInputMessage(confident_system_message=system_message))
            if user_message:
                input_messages.append(ConfidentLlmInputMessage(confident_user_message=user_message))
            if assistant_input_message:
                input_messages.append(ConfidentLlmInputMessage(confident_assistant_message=assistant_input_message))
            
            # Process output message
            output_message = attributes.get("gen_ai.assistant.message")
            llm_output = ConfidentLlmOutput(confident_assistant_message=output_message) if output_message else None
            
            # Create the ConfidentGenAiOperationSpan
            return ConfidentLlmSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                name=span.name,
                start_time_unix_nano=start_time_unix_nano,
                end_time_unix_nano=end_time_unix_nano,
                status_code=status_code,
                status_message=status_message,
                span_type=span_type,
                confident_request_model=model,
                confident_llm_input_messages=input_messages if input_messages else None,
                confident_llm_output=llm_output,
                confident_usage_input_tokens=input_tokens,
                confident_usage_output_tokens=output_tokens
            )

        
    def convert_confident_genai_operation_spans_to_base_api_spans(self, spans: List[BaseConfidentGenAiOperationSpan]) -> List[BaseApiSpan]:
        """
        Convert a list of OpenTelemetry GenAI Operation spans to BaseApiSpan objects.
        
        Args:
            spans: List of ConfidentGenAiOperationSpan objects from OpenTelemetry
            
        Returns:
            List of converted BaseApiSpan objects
        """
        api_spans = []
        
        for span in spans:
            # Determine span type
            span_type = SpanApiType.BASE
            if isinstance(span, ConfidentLlmSpan):
                span_type = SpanApiType.LLM
            
            # Convert timestamps from unix nano to ISO format
            start_time = to_zod_compatible_iso(
                datetime.fromtimestamp(span.start_time_unix_nano / 1e9, tz=timezone.utc)
            ) if span.start_time_unix_nano else ""
            end_time = to_zod_compatible_iso(
                datetime.fromtimestamp(span.end_time_unix_nano / 1e9, tz=timezone.utc)
            ) if span.end_time_unix_nano else ""
            
            # Create base API span with common attributes
            api_span = BaseApiSpan(
                uuid=span.span_id,
                name=span.name,
                status=TraceSpanApiStatus.SUCCESS if span.status_code == "OK" else TraceSpanApiStatus.ERRORED,
                type=span_type,
                traceUuid=span.trace_id,
                parentUuid=span.parent_span_id,
                startTime=start_time,
                endTime=end_time,
                error=span.status_message if span.status_code != "OK" else None
            )
            
            # Add type-specific attributes
            if isinstance(span, ConfidentLlmSpan):
                api_span.model = span.confident_request_model
                
                # Handle input and output
                if span.confident_llm_input_messages:
                    if isinstance(span.confident_llm_input_messages, list):
                        messages = []
                        for msg in span.confident_llm_input_messages:
                            if msg.confident_system_message:
                                messages.append({"role": "system", "content": msg.confident_system_message})
                            elif msg.confident_user_message:
                                messages.append({"role": "user", "content": msg.confident_user_message})
                            elif msg.confident_assistant_message:
                                messages.append({"role": "assistant", "content": msg.confident_assistant_message})
                        api_span.input = messages
                    else:
                        # Single message
                        msg = span.confident_llm_input_messages
                        if msg.confident_system_message:
                            api_span.input = msg.confident_system_message
                        elif msg.confident_user_message:
                            api_span.input = msg.confident_user_message
                        elif msg.confident_assistant_message:
                            api_span.input = msg.confident_assistant_message
                
                # Set output
                if span.confident_llm_output and span.confident_llm_output.confident_assistant_message:
                    api_span.output = span.confident_llm_output.confident_assistant_message
                
                # Set token counts
                api_span.input_token_count = span.confident_usage_input_tokens
                api_span.output_token_count = span.confident_usage_output_tokens
            
            api_spans.append(api_span)
        
        return api_spans

    def aggregate_base_api_spans_to_traces(self, spans: List[BaseApiSpan], environment: Optional[str] = "development" ) -> List[TraceApi]:
        # TODO: decide how to fetch environment from span attributes
        """
        Aggregate BaseApiSpan objects into TraceApi objects grouped by trace UUID.
        
        Args:
            spans: List of BaseApiSpan objects
            
        Returns:
            List of TraceApi objects, each representing a complete trace
        """
        # Group spans by trace UUID
        traces_dict = {}
        
        for span in spans:
            trace_uuid = span.trace_uuid
            if trace_uuid not in traces_dict:
                traces_dict[trace_uuid] = {
                    'baseSpans': [],
                    'agentSpans': [],
                    'llmSpans': [],
                    'retrieverSpans': [],
                    'toolSpans': [],
                    'metadata': None,
                    'tags': None,
                    'startTime': None,
                    'endTime': None
                }
            
            # Add span to appropriate list based on type
            if span.type == SpanApiType.AGENT.value:
                traces_dict[trace_uuid]['agentSpans'].append(span)
            elif span.type == SpanApiType.LLM.value:
                traces_dict[trace_uuid]['llmSpans'].append(span)
            elif span.type == SpanApiType.RETRIEVER.value:
                traces_dict[trace_uuid]['retrieverSpans'].append(span)
            elif span.type == SpanApiType.TOOL.value:
                traces_dict[trace_uuid]['toolSpans'].append(span)
            else:  # BASE type or any other type
                traces_dict[trace_uuid]['baseSpans'].append(span)
            
            # Track earliest start time and latest end time for the trace
            span_start_time = span.start_time
            span_end_time = span.end_time
            
            if span_start_time:
                # Parse ISO string to datetime for comparison
                span_start_dt = datetime.fromisoformat(span_start_time.replace('Z', '+00:00'))
                current_start_dt = datetime.fromisoformat(traces_dict[trace_uuid]['startTime'].replace('Z', '+00:00')) if traces_dict[trace_uuid]['startTime'] else None
                
                if not current_start_dt or span_start_dt < current_start_dt:
                    traces_dict[trace_uuid]['startTime'] = span_start_time
            
            if span_end_time:
                # Parse ISO string to datetime for comparison
                span_end_dt = datetime.fromisoformat(span_end_time.replace('Z', '+00:00'))
                current_end_dt = datetime.fromisoformat(traces_dict[trace_uuid]['endTime'].replace('Z', '+00:00')) if traces_dict[trace_uuid]['endTime'] else None
                
                if not current_end_dt or span_end_dt > current_end_dt:
                    traces_dict[trace_uuid]['endTime'] = span_end_time
        
        # Create TraceApi objects from the grouped spans
        trace_apis = []
        for trace_uuid, trace_data in traces_dict.items():
            trace_api = TraceApi(
                uuid=trace_uuid,
                baseSpans=trace_data['baseSpans'],
                agentSpans=trace_data['agentSpans'],
                llmSpans=trace_data['llmSpans'],
                retrieverSpans=trace_data['retrieverSpans'],
                toolSpans=trace_data['toolSpans'],
                startTime=trace_data['startTime'],
                endTime=trace_data['endTime'],
                metadata=trace_data['metadata'],
                tags=trace_data['tags'],
                environment=environment
            )
            trace_apis.append(trace_api)
        
        return trace_apis