from deepeval.tracing.types import Trace
from openai.types.responses.response_input_item_param import (
    FunctionCallOutput,
    Message,
)
from openai.types.responses.response_output_message_param import Content
from typing import Union, List, Optional
from openai.types.responses import (
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseInputContentParam,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputRefusal,
    EasyInputMessageParam,
    ResponseOutputMessage,
    ResponseOutputItem,
    ResponseOutputText,
)

from deepeval.tracing.types import (
    AgentSpan,
    ToolSpan,
    BaseSpan,
    LlmSpan,
)
import json

from deepeval.tracing.utils import make_json_serializable

try:
    from agents import MCPListToolsSpanData
    from agents.tracing.span_data import (
        AgentSpanData,
        FunctionSpanData,
        GenerationSpanData,
        ResponseSpanData,
        SpanData,
        HandoffSpanData,
        CustomSpanData,
        GuardrailSpanData,
    )

    openai_agents_available = True
except ImportError:
    openai_agents_available = False


def _check_openai_agents_available():
    if not openai_agents_available:
        raise ImportError(
            "openai-agents is required for this integration. Install it via your package manager"
        )


def update_span_properties(span: BaseSpan, span_data: "SpanData"):
    _check_openai_agents_available()
    # LLM Span
    if isinstance(span_data, ResponseSpanData):
        update_span_properties_from_response_span_data(span, span_data)
    elif isinstance(span_data, GenerationSpanData):
        update_span_properties_from_generation_span_data(span, span_data)
    # Tool Span
    elif isinstance(span_data, FunctionSpanData):
        update_span_properties_from_function_span_data(span, span_data)
    elif isinstance(span_data, MCPListToolsSpanData):
        update_span_properties_from_mcp_list_tool_span_data(span, span_data)
    # Agent Span
    elif isinstance(span_data, AgentSpanData):
        update_span_properties_from_agent_span_data(span, span_data)
    # Custom Span
    elif isinstance(span_data, HandoffSpanData):
        update_span_properties_from_handoff_span_data(span, span_data)
    elif isinstance(span_data, CustomSpanData):
        update_span_properties_from_custom_span_data(span, span_data)
    elif isinstance(span_data, GuardrailSpanData):
        update_span_properties_from_guardrail_span_data(span, span_data)


########################################################
### LLM Span ###########################################
########################################################


def update_span_properties_from_response_span_data(
    span: LlmSpan,
    span_data: "ResponseSpanData",
):
    response = span_data.response
    if response is None:
        span.model = "NA"
        return
    # Extract usage tokens
    usage = response.usage
    cached_input_tokens = None
    ouptut_reasoning_tokens = None
    if usage:
        output_tokens = usage.output_tokens
        input_tokens = usage.input_tokens
        cached_input_tokens = usage.input_tokens_details.cached_tokens
        ouptut_reasoning_tokens = usage.output_tokens_details.reasoning_tokens
    # Get input and output
    input = parse_response_input(
        span_data.input, span_data.response.instructions
    )
    raw_output = parse_response_output(response.output)
    output = (
        raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
    )
    # Update Span
    metadata = {
        "cached_input_tokens": cached_input_tokens,
        "ouptut_reasoning_tokens": ouptut_reasoning_tokens,
    }
    span.input_token_count = input_tokens
    span.output_token_count = output_tokens
    span.metadata = metadata
    span.model = "NA" if response.model is None else str(response.model)
    span.input = input
    span.output = output
    span.name = "LLM Generation"
    response_dict = response.model_dump(exclude_none=True, mode="json")
    span.metadata["invocation_params"] = {
        k: v
        for k, v in response_dict.items()
        if k
        in (
            "max_output_tokens",
            "parallel_tool_calls",
            "reasoning",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "truncation",
        )
    }


def update_span_properties_from_generation_span_data(
    span: LlmSpan,
    generation_span_data: "GenerationSpanData",
):
    # Extract usage tokens
    usage = generation_span_data.usage
    if usage:
        output_tokens = usage.get("output_tokens")
        input_tokens = usage.get("input_tokens")
    # Get input and output
    input = generation_span_data.input
    raw_output = generation_span_data.output
    output = (
        raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
    )
    # Update span
    span.input_token_count = input_tokens
    span.output_token_count = output_tokens
    span.model = generation_span_data.model or "NA"
    span.input = input
    span.output = output
    span.name = "LLM Generation"
    span.metadata["invocation_params"] = {
        "model_config": make_json_serializable(
            generation_span_data.model_config
        ),
    }


########################################################
### Tool Span ##########################################
########################################################


def update_span_properties_from_function_span_data(
    span: ToolSpan,
    function_span_data: "FunctionSpanData",
):
    # Update Span
    span.input = json.loads(function_span_data.input) or {
        "input": function_span_data.input
    }
    span.output = function_span_data.output
    span.name = (
        "Function tool: " + function_span_data.name
        if function_span_data.name
        else "Function tool"
    )
    span.description = "Function tool"


def update_span_properties_from_mcp_list_tool_span_data(
    span: ToolSpan,
    mcp_list_tool_span_data: "MCPListToolsSpanData",
):
    # Update Span
    span.input = None
    span.output = mcp_list_tool_span_data.result
    span.name = (
        "MCP tool: " + mcp_list_tool_span_data.server
        if mcp_list_tool_span_data.server
        else "MCP tool"
    )
    span.description = "MCP tool"


########################################################
### Agent Span #########################################
########################################################


def update_span_properties_from_agent_span_data(
    span: AgentSpan, agent_span_data: "AgentSpanData"
):
    # Update Span
    metadata = {}
    span.agent_handoffs = agent_span_data.handoffs
    span.available_tools = agent_span_data.tools
    span.name = agent_span_data.name
    if agent_span_data.output_type:
        metadata["output_type"] = agent_span_data.output_type
    span.metadata = metadata


########################################################
### Custom Span #######################################
########################################################


def update_span_properties_from_handoff_span_data(
    span: AgentSpan, handoff_span_data: "HandoffSpanData"
):
    # Update Span
    metadata = {
        "from_agent": handoff_span_data.from_agent,
        "to_agent": handoff_span_data.to_agent,
    }
    span.name = "Handoff → " + handoff_span_data.to_agent
    span.metadata = metadata
    span.input = None
    span.output = None


def update_span_properties_from_custom_span_data(
    span: BaseSpan, custom_span_data: "CustomSpanData"
):
    # Update Span
    span.name = custom_span_data.name
    span.metadata = {"data": custom_span_data.data}


def update_span_properties_from_guardrail_span_data(
    span: BaseSpan, guardrail_span_data: "GuardrailSpanData"
):
    # Update Span
    span.name = "Guardrail: " + guardrail_span_data.name
    span.metadata = {
        "data": guardrail_span_data.triggered,
        "type": guardrail_span_data.type,
    }


########################################################
### Parse Input Utils ##################################
########################################################


def parse_response_input(
    input: Union[str, List[ResponseInputItemParam]],
    instructions: Optional[Union[str, List[ResponseInputItemParam]]] = None,
):

    processed_input = []

    if isinstance(input, str) and isinstance(instructions, str):
        return [
            {"type": "message", "role": "system", "content": instructions},
            {"type": "message", "role": "user", "content": input},
        ]
    elif isinstance(input, list) and isinstance(instructions, list):
        input = instructions + input
    elif isinstance(input, list) and isinstance(instructions, str):
        processed_input += [
            {"type": "message", "role": "system", "content": instructions}
        ]
    elif isinstance(input, str) and isinstance(instructions, list):
        processed_input += [
            {"type": "message", "role": "user", "content": input}
        ]
        input = instructions

    for item in input:
        if "type" not in item:
            if "role" in item and "content" in item:
                processed_input.append(
                    {
                        "type": "message",
                        "role": item["role"],
                        "content": item["content"],
                    }
                )
        elif item["type"] == "message":
            parsed_message = parse_message_param(item)
            if parsed_message:
                processed_input.append(parsed_message)
        elif item["type"] == "function_call":
            processed_input.append(parse_function_tool_call_param(item))
        elif item["type"] == "function_call_output":
            processed_input.append(parse_function_call_output(item))
    return processed_input if processed_input else None


def parse_message_param(
    message: Union[
        EasyInputMessageParam,
        Message,
        ResponseOutputMessageParam,
    ],
):
    role = message["role"]
    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    elif isinstance(content, List):
        return {"role": role, "content": parse_message_content_list(content)}
    else:
        return None


def parse_message_content_list(
    content_list: List[Union[ResponseInputContentParam, Content]],
):
    processed_content_list = []
    for item in content_list:
        if item["type"] == "input_text" or item["type"] == "output_text":
            processed_content_list.append(
                {"type": "text", "text": item["text"]}
            )
        elif item["type"] == "input_image":
            # TODO
            ...
        elif item["type"] == "input_file":
            # TODO
            ...
        elif item["type"] == "refusal":
            processed_content_list.append(
                {"type": "refusal", "refusal": item["refusal"]}
            )
    return processed_content_list if processed_content_list else None


def parse_function_tool_call_param(
    tool_call_param: ResponseFunctionToolCallParam,
):
    return {
        "call_id": tool_call_param["call_id"],
        "name": tool_call_param["name"],
        "arguments": tool_call_param["arguments"],
    }


def parse_function_call_output(
    function_call_output: FunctionCallOutput,
):
    return {
        "role": "tool",
        "call_id": function_call_output["call_id"],
        "output": function_call_output["output"],
    }


########################################################
### Parse Output Utils ##################################
########################################################


def parse_response_output(response: List[ResponseOutputItem]):
    processed_output = []
    for item in response:
        if item.type == "message":
            message = parse_message(item)
            if isinstance(message, str):
                processed_output.append(message)
            elif isinstance(message, list):
                processed_output.extend(message)
        elif item.type == "function_call":
            processed_output.append(parse_function_call(item))
    if len(processed_output) == 1:
        return processed_output[0]
    return processed_output if processed_output else None


def parse_message(
    message: ResponseOutputMessage,
) -> Union[str, List[str]]:
    processed_content = []
    for item in message.content:
        if isinstance(item, ResponseOutputText):
            processed_content.append(item.text)
        elif isinstance(item, ResponseOutputRefusal):
            processed_content.append(item.refusal)
    if len(processed_content) == 1:
        return processed_content[0]
    return processed_content if processed_content else None


def parse_function_call(
    function_call: ResponseFunctionToolCall,
):
    return {
        "call_id": function_call.call_id,
        "name": function_call.name,
        "arguments": function_call.arguments,
    }


def update_trace_properties_from_span_data(
    trace: Trace,
    span_data: Union["ResponseSpanData", "GenerationSpanData"],
):
    if isinstance(span_data, ResponseSpanData):
        if not trace.input:
            trace.input = parse_response_input(
                span_data.input, span_data.response.instructions
            )
        raw_output = parse_response_output(span_data.response.output)
        output = (
            raw_output
            if isinstance(raw_output, str)
            else json.dumps(raw_output)
        )
        trace.output = output

    elif isinstance(span_data, GenerationSpanData):
        if not trace.input:
            trace.input = span_data.input
        raw_output = span_data.output
        output = (
            raw_output
            if isinstance(raw_output, str)
            else json.dumps(raw_output)
        )
        trace.output = output
