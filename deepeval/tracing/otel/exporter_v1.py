import json
import typing
from dataclasses import dataclass
from typing import List, Optional
from opentelemetry.trace.status import StatusCode
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, ReadableSpan
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing import trace_manager
from deepeval.tracing.attributes import AgentAttributes, LlmAttributes, RetrieverAttributes, ToolAttributes
from deepeval.tracing.types import AgentSpan, BaseSpan, LlmSpan, RetrieverSpan, ToolSpan, TraceSpanStatus
from deepeval.tracing.otel.utils import to_hex_string, set_trace_time, validate_llm_test_case_data
import deepeval
from deepeval.tracing import perf_epoch_bridge as peb
from deepeval.test_case import LLMTestCase, ToolCall
from pydantic import BaseModel, ValidationError
from deepeval.feedback.feedback import Feedback

class TraceAttributes(BaseModel):
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class BaseSpanWrapper:
    base_span: BaseSpan
    trace_attributes: Optional[TraceAttributes] = None

class ConfidentSpanExporterV1(SpanExporter):
    
    def __init__(self, api_key: Optional[str] = None):
        capture_tracing_integration("deepeval.tracing.otel.exporter_v1")
        peb.init_clock_bridge()
        
        if api_key:
            deepeval.login_with_confident_api_key(api_key) # TODO: send api keys dynamically to get it compatible with the collector framework
        
        super().__init__()

    def export(
            self,
            spans: typing.Sequence[ReadableSpan],
            timeout_millis: int = 30000
        ) -> SpanExportResult:
        
        spans_wrappers_list: List[BaseSpanWrapper] = []
        
        for span in spans:

            # confugarion are attached to the resource attributes
            resource_attributes = span.resource.attributes
            environment = resource_attributes.get("confident.environment")
            if environment and isinstance(environment, str):
                trace_manager.configure(environment=environment)
            
            sampling_rate = resource_attributes.get("confident.sampling_rate")
            if sampling_rate and isinstance(sampling_rate, float):
                trace_manager.configure(sampling_rate=sampling_rate)
            
            spans_wrappers_list.append(self._convert_readable_span_to_base_span(span))
        
        # list starts from root span
        for base_span_wrapper in reversed(spans_wrappers_list):
            current_trace = trace_manager.get_trace_by_uuid(base_span_wrapper.base_span.trace_uuid)
            if not current_trace:
                current_trace = trace_manager.start_new_trace(trace_uuid=base_span_wrapper.base_span.trace_uuid)
            
            # set the trace attributes
            if base_span_wrapper.trace_attributes:
                current_trace.name = base_span_wrapper.trace_attributes.name
                current_trace.tags = base_span_wrapper.trace_attributes.tags
                current_trace.thread_id = base_span_wrapper.trace_attributes.thread_id
                current_trace.user_id = base_span_wrapper.trace_attributes.user_id
                
            trace_manager.add_span(base_span_wrapper.base_span)
            trace_manager.add_span_to_trace(base_span_wrapper.base_span)
            # no removing span because it can be parent of other spans


        # safely end all active traces
        active_traces_keys = list(trace_manager.active_traces.keys())
        
        for trace_key in active_traces_keys:
            set_trace_time(trace_manager.get_trace_by_uuid(trace_key))
            trace_manager.end_trace(trace_key)

        trace_manager.clear_traces()
        
        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
    
    def _convert_readable_span_to_base_span(self, span: ReadableSpan) -> BaseSpanWrapper:
        base_span = None
        try:
            base_span = self._prepare_boilerplate_base_span(span)
        except Exception as e:
            print(f"Error converting span: {e}")

        # fallback to base span with default values
        if not base_span:
            input = span.attributes.get("confident.span.input")
            output = span.attributes.get("confident.span.output")
            
            base_span = BaseSpan(
                uuid=to_hex_string(span.context.span_id, 16),
                status=TraceSpanStatus.ERRORED if span.status.status_code == StatusCode.ERROR else TraceSpanStatus.SUCCESS,
                children=[],
                trace_uuid=to_hex_string(span.context.trace_id, 32),
                parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None,
                start_time=peb.epoch_nanos_to_perf_seconds(span.start_time),
                end_time=peb.epoch_nanos_to_perf_seconds(span.end_time),
                input=input,
                output=output
            )

        # extract error
        error = span.attributes.get("confident.span.error")

        # validate feedback
        feedback_json_str = span.attributes.get("confident.span.feedback")
        feedback = None
        if feedback_json_str:
            try:
                feedback = Feedback.model_validate_json(feedback_json_str)
            except ValidationError as err:
                print(f"Error converting feedback: {err}")

        # extract metric collection
        metric_collection = span.attributes.get("confident.span.metric_collection")
        if not isinstance(metric_collection, str):
            metric_collection = None
        
        # extract llm test case attributes (except additional_metadata, tools_called, expected_tools)
        test_case_input = span.attributes.get("confident.span.llm_test_case.input")
        test_case_actual_output = span.attributes.get("confident.span.llm_test_case.actual_output")
        test_case_expected_output = span.attributes.get("confident.span.llm_test_case.expected_output")
        test_case_context = span.attributes.get("confident.span.llm_test_case.context")
        test_case_retrieval_context = span.attributes.get("confident.span.llm_test_case.retrieval_context")

        # validate list of strings for tool calls and expected tools
        test_case_tools_called_attr = span.attributes.get("confident.span.llm_test_case.tools_called")
        test_case_expected_tools_attr = span.attributes.get("confident.span.llm_test_case.expected_tools")

        tools_called: List[ToolCall] = []
        expected_tools: List[ToolCall] = []

        if test_case_tools_called_attr and isinstance(test_case_tools_called_attr, list):
            for tool_call_json_str in test_case_tools_called_attr:
                if isinstance(tool_call_json_str, str):
                    try:
                        tools_called.append(ToolCall.model_validate_json(tool_call_json_str))
                    except ValidationError as err:
                        print(f"Error converting tool call: {err}")
        
        if test_case_expected_tools_attr and isinstance(test_case_expected_tools_attr, list):
            for tool_call_json_str in test_case_expected_tools_attr:
                if isinstance(tool_call_json_str, str):
                    try:
                        expected_tools.append(ToolCall.model_validate_json(tool_call_json_str))
                    except ValidationError as err:
                        print(f"Error converting expected tool call: {err}")
        
        llm_test_case = None
        if test_case_input and test_case_actual_output:
            try:
                validate_llm_test_case_data(test_case_input, test_case_actual_output, test_case_expected_output, test_case_context, test_case_retrieval_context, tools_called, expected_tools)
                
                llm_test_case = LLMTestCase(
                    input=test_case_input,
                    actual_output=test_case_actual_output,
                    expected_output=test_case_expected_output,
                    context=test_case_context,
                    retrieval_context=test_case_retrieval_context,
                    tools_called=tools_called,
                    expected_tools=expected_tools
                )
            except Exception as e:
                print(f"Invalid LLMTestCase data: {e}")
        
        base_span.parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None
        base_span.name=span.name
        base_span.metadata=json.loads(span.to_json())
        base_span.error=error
        base_span.llm_test_case=llm_test_case
        base_span.metric_collection=metric_collection
        base_span.feedback=feedback

        # extract trace attributes
        trace_attributes = None
        trace_attr_json_str = span.attributes.get("confident.trace.attributes")
        if trace_attr_json_str:
            try:
                trace_attributes = TraceAttributes.model_validate_json(trace_attr_json_str)
            except ValidationError as err:
                print(f"Error converting trace attributes: {err}")

        base_span_wrapper = BaseSpanWrapper(
            base_span=base_span,
            trace_attributes=trace_attributes
        )

        return base_span_wrapper
    
    def _prepare_boilerplate_base_span(self, span: ReadableSpan) -> Optional[BaseSpan]:
        span_type = span.attributes.get("confident.span.type", "base")
        
        # required fields
        uuid=to_hex_string(span.context.span_id, 16)
        status=TraceSpanStatus.ERRORED if span.status.status_code == StatusCode.ERROR else TraceSpanStatus.SUCCESS
        children=[]
        trace_uuid=to_hex_string(span.context.trace_id, 32)
        parent_uuid=to_hex_string(span.parent.span_id, 16) if span.parent else None
        start_time=peb.epoch_nanos_to_perf_seconds(span.start_time)
        end_time=peb.epoch_nanos_to_perf_seconds(span.end_time)
        
        if span_type == "llm":
            model = span.attributes.get("confident.span.model")
            cost_per_input_token = span.attributes.get("confident.span.cost_per_input_token")
            cost_per_output_token = span.attributes.get("confident.span.cost_per_output_token")

            llm_span = LlmSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,

                # llm span attributes
                model=model,
                cost_per_input_token=cost_per_input_token,
                cost_per_output_token=cost_per_output_token
            )

            # set attributes
            input = span.attributes.get("confident.span.attributes.input")
            output = span.attributes.get("confident.span.attributes.output")
            prompt = span.attributes.get("confident.span.attributes.prompt")
            input_token_count = span.attributes.get("confident.span.attributes.input_token_count")
            output_token_count = span.attributes.get("confident.span.attributes.output_token_count")

            try:
                llm_span.set_attributes(LlmAttributes(
                    input=input,
                    output=output,
                    prompt=prompt,
                    input_token_count=input_token_count,
                    output_token_count=output_token_count
                ))
            except Exception as e:
                print(f"Error setting llm span attributes: {e}")

            return llm_span
        
        elif span_type == "agent":
            name = span.attributes.get("confident.span.name")
            available_tools = span.attributes.get("confident.span.available_tools")
            agent_handoffs = span.attributes.get("confident.span.agent_handoffs")

            agent_span = AgentSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,

                # agent span attributes
                name=name if name else "",
                available_tools=available_tools if available_tools else [],
                agent_handoffs=agent_handoffs if agent_handoffs else []
            )

            # set attributes
            input = span.attributes.get("confident.span.attributes.input")
            output = span.attributes.get("confident.span.attributes.output")
            
            try:
                agent_span.set_attributes(AgentAttributes(
                    input=input,
                    output=output
                ))
            except Exception as e:
                print(f"Error setting agent span attributes: {e}")

            return agent_span
        
        elif span_type == "retriever":
            embedder = span.attributes.get("confident.span.attributes.embedder")

            retriever_span = RetrieverSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,

                # retriever span attributes
                embedder=embedder if embedder else "",
            )

            # set attributes
            embedding_input = span.attributes.get("confident.span.attributes.embedding_input")
            retrieval_context = span.attributes.get("confident.span.attributes.retrieval_context")
            top_k = span.attributes.get("confident.span.attributes.top_k")
            chunk_size = span.attributes.get("confident.span.attributes.chunk_size")

            try:
                retriever_span.set_attributes(RetrieverAttributes(
                    embedding_input=embedding_input,
                    retrieval_context=retrieval_context,
                    top_k=top_k,
                    chunk_size=chunk_size
                ))
            except Exception as e:
                print(f"Error setting retriever span attributes: {e}")

            return retriever_span

        elif span_type == "tool":
            name = span.attributes.get("confident.span.name")
            description = span.attributes.get("confident.span.description")
    
            tool_span = ToolSpan(
                uuid=uuid,
                status=status,
                children=children,
                trace_uuid=trace_uuid,
                parent_uuid=parent_uuid,
                start_time=start_time,
                end_time=end_time,

                # tool span attributes
                name=name if name else "",
                description=description
            )

            # set attributes
            input_parameters = span.attributes.get("confident.span.attributes.input_parameters")
            output = span.attributes.get("confident.span.attributes.output")

            try:
                tool_span.set_attributes(ToolAttributes(
                    input_parameters=input_parameters,
                    output=output
                ))
            except Exception as e:
                print(f"Error setting tool span attributes: {e}")

            return tool_span
        
        # if span type is not supported, return None
        return None

