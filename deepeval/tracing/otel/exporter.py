from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from typing import Dict, List, Optional
from datetime import datetime, timezone
from deepeval.tracing.api import TraceSpanApiStatus
from deepeval.tracing.otel.utils import to_hex_string
from deepeval.tracing.tracing import (
    BaseApiSpan,
    SpanApiType,
    TraceApi,
    to_zod_compatible_iso,
    trace_manager,
)
import typing
import json


class ConfidentSpanExporter(SpanExporter):
    def __init__(self):
        self.trace_manager = trace_manager
        super().__init__()

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        base_api_spans = [self.convert_to_base_api_span(span) for span in spans]
        traces = self.aggregate_base_api_spans_to_traces(base_api_spans)
        for trace in traces:
            self.trace_manager.post_trace_api(trace)

        return SpanExportResult.SUCCESS

    def shutdown(self):
        # TODO: implement
        pass

    def force_flush(self, timeout_millis=30000):
        self.trace_manager.flush_traces()

    def convert_to_base_api_span(
        self, span: ReadableSpan
    ) -> Optional[BaseApiSpan]:
        # if span is not relevant confident spans, return None
        if not (span.context.trace_id and span.context.span_id):
            return None

        uuid = to_hex_string(span.context.span_id, 16)  # uuid
        name = span.name if span.name else None  # name
        # default to success, if error, set to errored
        status = (
            TraceSpanApiStatus.ERRORED
            if span.status.status_code == StatusCode.ERROR
            else TraceSpanApiStatus.SUCCESS
        )
        trace_uuid = to_hex_string(span.context.trace_id, 32)  # trace_uuid
        parent_uuid = (
            to_hex_string(span.parent.span_id, 16) if span.parent else None
        )
        start_time = to_zod_compatible_iso(
            datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc)
        )
        end_time = to_zod_compatible_iso(
            datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc)
        )
        error = (
            span.status.description
            if status == TraceSpanApiStatus.ERRORED
            else None
        )
        # span_test_case
        # metrics
        # metrics_data

        base_api_span = BaseApiSpan(
            uuid=uuid,
            name=name,
            status=status,
            type=SpanApiType.BASE,
            traceUuid=trace_uuid,
            parentUuid=parent_uuid,
            startTime=start_time,
            endTime=end_time,
            error=error,
        )

        _attributes = span.attributes if span.attributes else {}

        if _attributes.get("gen_ai.operation.name") in [
            "chat",
            "text_completion",
            "generate_content",
        ]:
            _llm_attributes = self._extract_llm_attributes(span, res={})
            base_api_span.type = SpanApiType.LLM
            base_api_span.input = _llm_attributes.get("input")
            base_api_span.output = _llm_attributes.get("output")
            base_api_span.model = _llm_attributes.get("model")
            base_api_span.input_token_count = _llm_attributes.get(
                "input_token_count"
            )
            base_api_span.output_token_count = _llm_attributes.get(
                "output_token_count"
            )
            base_api_span.cost_per_input_token = _llm_attributes.get(
                "cost_per_input_token"
            )
            base_api_span.cost_per_output_token = _llm_attributes.get(
                "cost_per_output_token"
            )

        elif _attributes.get("gen_ai.operation.name") in ["execute_tool"]:
            _tool_attributes = self._extract_tool_attributes(span, res={})
            base_api_span.type = SpanApiType.TOOL
            base_api_span.input = _tool_attributes.get("input")
            base_api_span.output = _tool_attributes.get("output")
            base_api_span.description = _tool_attributes.get("description")

        elif _attributes.get("gen_ai.operation.name") in [
            "create_agent",
            "invoke_agent",
        ]:
            _agent_attributes = self._extract_agent_attributes(span, res={})
            base_api_span.type = SpanApiType.AGENT
            base_api_span.input = _agent_attributes.get("input")
            base_api_span.output = _agent_attributes.get("output")
            base_api_span.available_tools = _agent_attributes.get(
                "available_tools"
            )
            base_api_span.agent_handoffs = _agent_attributes.get(
                "agent_handoffs"
            )

        elif _attributes.get("gen_ai.operation.name") in ["embeddings"]:
            _retriever_attributes = self._extract_retriever_attributes(
                span, res={}
            )
            base_api_span.type = SpanApiType.RETRIEVER
            base_api_span.input = _retriever_attributes.get("input")
            base_api_span.output = _retriever_attributes.get("output")
            base_api_span.top_k = _retriever_attributes.get("top_k")
            base_api_span.chunk_size = _retriever_attributes.get("chunk_size")

        # dump span to metadata dict
        # todo: dump only relevant attributes
        base_api_span.metadata = {"span": json.loads(span.to_json())}
        return base_api_span

    def _extract_llm_attributes(self, span: ReadableSpan, res: Dict) -> Dict:
        res["input"] = []
        for event in span.events:
            # input
            if event.name in ["gen_ai.system.message", "gen_ai.user.message"]:
                attributes = event.attributes
                res["input"].append(
                    {
                        "role": event.name,
                        "content": (
                            dict(attributes) if attributes is not None else {}
                        ),
                    }
                )
            # output
            if event.name in [
                "gen_ai.assistant.message",
                "gen_ai.choice",
                "gen_ai.tool.message",
            ]:
                res["output"] = str(event.attributes)

        model = span.attributes.get("gen_ai.request.model")
        res["model"] = str(model) if model is not None else None

        input_tokens = span.attributes.get("gen_ai.usage.input_tokens")
        res["input_token_count"] = (
            int(input_tokens) if input_tokens is not None else None
        )

        output_tokens = span.attributes.get("gen_ai.usage.output_tokens")
        res["output_token_count"] = (
            int(output_tokens) if output_tokens is not None else None
        )

        cost_per_input = span.attributes.get(
            "confident.llm.cost_per_input_token"
        )
        res["cost_per_input_token"] = (
            float(cost_per_input) if cost_per_input is not None else None
        )

        cost_per_output = span.attributes.get(
            "confident.llm.cost_per_output_token"
        )
        res["cost_per_output_token"] = (
            float(cost_per_output) if cost_per_output is not None else None
        )

        return res

    def _extract_tool_attributes(self, span: ReadableSpan, res: Dict) -> Dict:
        for event in span.events:
            # input
            if event.name in ["confident.tool.input"]:
                res["input"] = (
                    dict(event.attributes)
                    if event.attributes is not None
                    else {}
                )
            # output
            if event.name in ["confident.tool.output"]:
                res["output"] = (
                    dict(event.attributes)
                    if event.attributes is not None
                    else {}
                )

        description = span.attributes.get("gen_ai.tool.description")
        res["description"] = (
            str(description) if description is not None else None
        )

        return res

    def _extract_agent_attributes(self, span: ReadableSpan, res: Dict) -> Dict:
        for event in span.events:
            # input
            if event.name in ["confident.agent.input"]:
                res["input"] = (
                    dict(event.attributes)
                    if event.attributes is not None
                    else {}
                )
            # output
            if event.name in ["confident.agent.output"]:
                res["output"] = (
                    dict(event.attributes)
                    if event.attributes is not None
                    else {}
                )

        available_tools = span.attributes.get("confident.agent.available_tools")
        if isinstance(available_tools, tuple):
            res["available_tools"] = [str(tool) for tool in available_tools]

        agent_handoffs = span.attributes.get("confident.agent.agent_handoffs")
        if isinstance(agent_handoffs, tuple):
            res["agent_handoffs"] = [str(handoff) for handoff in agent_handoffs]

        return res

    def _extract_retriever_attributes(
        self, span: ReadableSpan, res: Dict
    ) -> Dict:
        for event in span.events:
            # input
            if event.name in ["confident.retriever.input"]:
                res["input"] = str(event.attributes)

            # output
            if event.name in ["confident.retriever.output"]:
                attributes = (
                    event.attributes
                )  # event attributes might not be a tuple (#TODO: check)
                if isinstance(attributes, tuple):
                    res["output"] = [str(attribute) for attribute in attributes]

        res["embedder"] = str(span.attributes.get("gen_ai.request.model"))
        top_k = span.attributes.get("confident.retriever.top_k")
        res["top_k"] = int(top_k) if top_k is not None else None
        chunk_size = span.attributes.get("confident.retriever.chunk_size")
        res["chunk_size"] = int(chunk_size) if chunk_size is not None else None
        return res

    def aggregate_base_api_spans_to_traces(
        self,
        spans: List[BaseApiSpan],
        environment: Optional[str] = "development",
    ) -> List[TraceApi]:
        traces_dict = {}

        for span in spans:
            trace_uuid = span.trace_uuid
            if trace_uuid not in traces_dict:
                traces_dict[trace_uuid] = {
                    "baseSpans": [],
                    "agentSpans": [],
                    "llmSpans": [],
                    "retrieverSpans": [],
                    "toolSpans": [],
                    "metadata": None,
                    "tags": None,
                    "startTime": None,
                    "endTime": None,
                }

            # Add span to appropriate list based on type
            if span.type == SpanApiType.AGENT:
                traces_dict[trace_uuid]["agentSpans"].append(span)
            elif span.type == SpanApiType.LLM:
                traces_dict[trace_uuid]["llmSpans"].append(span)
            elif span.type == SpanApiType.RETRIEVER:
                traces_dict[trace_uuid]["retrieverSpans"].append(span)
            elif span.type == SpanApiType.TOOL:
                traces_dict[trace_uuid]["toolSpans"].append(span)
            else:  # BASE type or any other type
                traces_dict[trace_uuid]["baseSpans"].append(span)

            # Track earliest start time and latest end time for the trace
            span_start_time = span.start_time
            span_end_time = span.end_time

            if span_start_time:
                # Parse ISO string to datetime for comparison
                span_start_dt = datetime.fromisoformat(
                    span_start_time.replace("Z", "+00:00")
                )
                current_start_dt = (
                    datetime.fromisoformat(
                        traces_dict[trace_uuid]["startTime"].replace(
                            "Z", "+00:00"
                        )
                    )
                    if traces_dict[trace_uuid]["startTime"]
                    else None
                )

                if not current_start_dt or span_start_dt < current_start_dt:
                    traces_dict[trace_uuid]["startTime"] = span_start_time

            if span_end_time:
                # Parse ISO string to datetime for comparison
                span_end_dt = datetime.fromisoformat(
                    span_end_time.replace("Z", "+00:00")
                )
                current_end_dt = (
                    datetime.fromisoformat(
                        traces_dict[trace_uuid]["endTime"].replace(
                            "Z", "+00:00"
                        )
                    )
                    if traces_dict[trace_uuid]["endTime"]
                    else None
                )

                if not current_end_dt or span_end_dt > current_end_dt:
                    traces_dict[trace_uuid]["endTime"] = span_end_time

        # Create TraceApi objects from the grouped spans
        trace_apis = []
        for trace_uuid, trace_data in traces_dict.items():
            trace_api = TraceApi(
                uuid=trace_uuid,
                baseSpans=trace_data["baseSpans"],
                agentSpans=trace_data["agentSpans"],
                llmSpans=trace_data["llmSpans"],
                retrieverSpans=trace_data["retrieverSpans"],
                toolSpans=trace_data["toolSpans"],
                startTime=trace_data["startTime"],
                endTime=trace_data["endTime"],
                metadata=trace_data["metadata"],
                tags=trace_data["tags"],
                environment=environment,
            )
            trace_apis.append(trace_api)

        return trace_apis
