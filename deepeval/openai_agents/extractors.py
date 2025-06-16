from openai.types.responses import Response
from typing import Optional, Union, List

from deepeval.tracing.attributes import ToolAttributes, LlmAttributes
from deepeval.prompt.prompt import Prompt
from deepeval.tracing.types import (
    BaseSpan,
    LlmSpan,
    ToolSpan,
    AgentSpan,
)

# check openai agents availability
try:
    from agents import MCPListToolsSpanData
    from agents.tracing.span_data import (
        AgentSpanData,
        FunctionSpanData,
        GenerationSpanData,
        ResponseSpanData,
        SpanData,
    )
    openai_agents_available = True
except ImportError:
    openai_agents_available = False

def _check_openai_agents_available():
    if not openai_agents_available:
        raise ImportError(
            "openai-agents is required for this integration. Install it via your package manager"
        )

def custom_update_span_attributes(span: BaseSpan, span_data: SpanData):
    _check_openai_agents_available()
    # LLM Span
    if isinstance(span_data, ResponseSpanData):
        update_attributes_from_response_span_data(span, span_data.response, span_data.input)
    elif isinstance(span_data, GenerationSpanData):
        udpate_attributes_from_generation_span_data(span, span_data)
    # Tool Span
    elif isinstance(span_data, FunctionSpanData):
        update_attributes_from_function_span_data(span, span_data)
    elif isinstance(span_data, MCPListToolsSpanData):
        update_attributes_from_mcp_list_tool_span_data(span, span_data)
    # Agent Span 
    elif isinstance(span_data, AgentSpanData):
        update_attributes_from_agent_span_data(span, span_data)

########################################################
### LLM Span ###########################################
########################################################

def update_attributes_from_response_span_data(
    span: LlmSpan, 
    response: Optional[Response] = None,
    input: Optional[Union[str, List]] = None 
):
    if response is None:
        span.model = "NA"

    # Extract prompt template
    prompt_template = response.instructions
    prompt = Prompt(template=prompt_template) if prompt_template else None
    # Extract usage tokens
    usage = response.usage
    if usage:
        output_tokens = usage.output_tokens
        input_tokens = usage.input_tokens
        cached_input_tokens= usage.input_tokens_details.cached_tokens
        ouptut_reasoning_tokens= usage.output_tokens_details.reasoning_tokens
    # Update Span
    llm_attributes=LlmAttributes(
        prompt=prompt,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        input=input,
        output=""
    )
    metadata = {
        "cached_input_tokens": cached_input_tokens,
        "ouptut_reasoning_tokens": ouptut_reasoning_tokens
    }
    span.set_attributes(llm_attributes)
    span.metadata = metadata
    span.model = "NA" if response.model is None else str(response.model)
    span.input = llm_attributes.input
    span.output = response.output

def udpate_attributes_from_generation_span_data(
    span: LlmSpan, 
    generation_span_data: GenerationSpanData,
):
    # Extract usage tokens
    usage =generation_span_data.usage
    if usage:
        output_tokens = usage.get("output_tokens")
        input_tokens = usage.get("input_tokens")
    # Update span
    llm_attributes=LlmAttributes(
        prompt=None,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        input=generation_span_data.input,
        output=generation_span_data.output
    )
    span.set_attributes(llm_attributes)
    span.model = generation_span_data.model or "NA"
    span.input = llm_attributes.input
    span.output = llm_attributes.output


########################################################
### Tool Span ##########################################
########################################################

def update_attributes_from_function_span_data(
    span: ToolSpan, 
    function_span_data: FunctionSpanData,
):
    # Update Span
    tool_attributes = ToolAttributes(
        input_parameters=function_span_data.input,
        output=function_span_data.output
    )
    span.set_attributes(tool_attributes)
    span.name = function_span_data.name
    span.description = "Function tool"

def update_attributes_from_mcp_list_tool_span_data(
    span: ToolSpan, 
    mcp_list_tool_span_data: MCPListToolsSpanData,
) :
    # Update Span
    tool_attributes = ToolAttributes(
        input_parameters=None,
        output=mcp_list_tool_span_data.result
    )
    span.set_attributes(tool_attributes)
    span.name = mcp_list_tool_span_data.server
    span.description = "MCP tool"


########################################################
### Agent Span #########################################
########################################################

def update_attributes_from_agent_span_data(
    span: AgentSpan, 
    agent_span_data: AgentSpanData
):
    # Update Span
    metadata={}
    span.agent_handoffs = agent_span_data.handoffs
    span.available_tools = agent_span_data.tools
    span.name = agent_span_data.name
    if agent_span_data.output_type:
        metadata["output_type"] = agent_span_data.output_type
    span.metadata = metadata
    span.input = None
    span.output = None