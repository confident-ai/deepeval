from openai.types.responses.response_input_item_param import (
    FunctionCallOutput,
    Message,
)
from openai.types.responses.response_output_message_param import Content
from typing import Union, List
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

from deepeval.tracing.attributes import ToolAttributes, LlmAttributes
from deepeval.prompt.prompt import Prompt
from deepeval.tracing.types import (
    AgentSpan,
    ToolSpan,
    BaseSpan,
    LlmSpan,
)
import json

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


def update_span_properties(span: BaseSpan, span_data: SpanData):
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
    span_data: ResponseSpanData,
):
    response = span_data.response
    if response is None:
        span.model = "NA"
        return
    # Extract prompt template
    prompt_template = response.instructions or None
    prompt = Prompt(template=prompt_template) if prompt_template else None
    # Extract usage tokens
    usage = response.usage
    if usage:
        output_tokens = usage.output_tokens
        input_tokens = usage.input_tokens
        cached_input_tokens = usage.input_tokens_details.cached_tokens
        ouptut_reasoning_tokens = usage.output_tokens_details.reasoning_tokens
    # Get input and output
    input = parse_response_input(span_data.input)
    raw_output = parse_response_output(response.output)
    output = (
        raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
    )
    # Update Span
    llm_attributes = LlmAttributes(
        prompt=prompt,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        input=input,
        output=output,
    )
    metadata = {
        "cached_input_tokens": cached_input_tokens,
        "ouptut_reasoning_tokens": ouptut_reasoning_tokens,
    }
    span.set_attributes(llm_attributes)
    span.metadata = metadata
    span.model = "NA" if response.model is None else str(response.model)
    span.input = input
    span.output = output
    span.name = "LLM Generation"


def update_span_properties_from_generation_span_data(
    span: LlmSpan,
    generation_span_data: GenerationSpanData,
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
    llm_attributes = LlmAttributes(
        prompt=None,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        input=input,
        output=output,
    )
    span.set_attributes(llm_attributes)
    span.model = generation_span_data.model or "NA"
    span.input = input
    span.output = output
    span.name = "LLM Generation"


########################################################
### Tool Span ##########################################
########################################################


def update_span_properties_from_function_span_data(
    span: ToolSpan,
    function_span_data: FunctionSpanData,
):
    # Update Span
    tool_attributes = ToolAttributes(
        input_parameters=json.loads(function_span_data.input)
        or {"input": function_span_data.input},
        output=function_span_data.output,
    )
    span.set_attributes(tool_attributes)
    span.name = (
        "Function tool: " + function_span_data.name
        if function_span_data.name
        else "Function tool"
    )
    span.description = "Function tool"


def update_span_properties_from_mcp_list_tool_span_data(
    span: ToolSpan,
    mcp_list_tool_span_data: MCPListToolsSpanData,
):
    # Update Span
    tool_attributes = ToolAttributes(
        input_parameters=None, output=mcp_list_tool_span_data.result
    )
    span.set_attributes(tool_attributes)
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
    span: AgentSpan, agent_span_data: AgentSpanData
):
    # Update Span
    metadata = {}
    span.agent_handoffs = agent_span_data.handoffs
    span.available_tools = agent_span_data.tools
    span.name = agent_span_data.name
    if agent_span_data.output_type:
        metadata["output_type"] = agent_span_data.output_type
    span.metadata = metadata
    span.input = None
    span.output = None


########################################################
### Custom Span #######################################
########################################################


def update_span_properties_from_handoff_span_data(
    span: AgentSpan, handoff_span_data: HandoffSpanData
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
    span: BaseSpan, custom_span_data: CustomSpanData
):
    # Update Span
    span.name = custom_span_data.name
    span.metadata = {"data": custom_span_data.data}


def update_span_properties_from_guardrail_span_data(
    span: BaseSpan, guardrail_span_data: GuardrailSpanData
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


def parse_response_input(input: Union[str, List[ResponseInputItemParam]]):
    if isinstance(input, str):
        return input
    processed_input = []
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
