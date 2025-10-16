import json
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from typing import Any, Union, Dict
from openai.types.responses import Response

from deepeval.test_case.llm_test_case import ToolCall
from deepeval.openai.utils import stringify_multimodal_content, render_messages
from deepeval.openai.types import InputParameters, OutputParameters
from deepeval.tracing.types import Message


def extract_input_parameters(
    is_completion: bool, kwargs: Dict[str, Any]
) -> InputParameters:
    if is_completion:
        return extract_input_parameters_from_completion(kwargs)
    else:
        return extract_input_parameters_from_response(kwargs)


def extract_input_parameters_from_completion(
    kwargs: Dict[str, Any],
) -> InputParameters:
    model = kwargs.get("model")
    messages = kwargs.get("messages") or []
    tools = kwargs.get("tools")
    tool_descriptions_map = (
        {
            tool["function"]["name"]: tool["function"]["description"]
            for tool in tools
        }
        if tools is not None
        else None
    )

    # extract first user input from messages
    input_arg = ""
    user_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            user_messages.append(content)
    if len(user_messages) > 0:
        input_arg = user_messages[0]
    
    # render messages
    messages = render_messages(messages)

    return InputParameters(
        model=model,
        input=stringify_multimodal_content(input_arg),
        messages=messages,
        tools=tools,
        tool_descriptions=tool_descriptions_map,
    )


def extract_input_parameters_from_response(
    kwargs: Dict[str, Any],
) -> InputParameters:
    model = kwargs.get("model")
    input_payload = kwargs.get("input")
    instructions = kwargs.get("instructions")
    tools = kwargs.get("tools")
    tool_descriptions = (
        {tool["name"]: tool["description"] for tool in tools}
        if tools is not None
        else None
    )
    messages = input_payload if isinstance(input_payload, list) else None
    return InputParameters(
        model=model,
        input=stringify_multimodal_content(input_payload),
        messages=messages,
        instructions=instructions,
        tools=tools,
        tool_descriptions=tool_descriptions,
    )


def extract_output_parameters(
    is_completion: bool,
    response: Union[ChatCompletion, ParsedChatCompletion, Response],
    input_parameters: InputParameters,
) -> OutputParameters:
    if is_completion:
        return extract_output_parameters_from_completion(
            response, input_parameters
        )
    else:
        return extract_output_parameters_from_response(
            response, input_parameters
        )


def extract_output_parameters_from_completion(
    completion: Union[ChatCompletion, ParsedChatCompletion],
    input_parameters: InputParameters,
) -> OutputParameters:
    output = str(completion.choices[0].message.content or "")
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    # Extract Tools Called
    tools_called = None
    openai_tool_calls = completion.choices[0].message.tool_calls
    if openai_tool_calls is not None:
        tools_called = []
        for tool_call in openai_tool_calls:
            tool_descriptions = input_parameters.tool_descriptions or {}
            tools_called.append(
                ToolCall(
                    name=tool_call.function.name,
                    input_parameters=json.loads(tool_call.function.arguments),
                    description=tool_descriptions.get(tool_call.function.name),
                )
            )
    
    # If output is empty and tool calls exist, format them as string
    if not output and tools_called:
        formatted_calls = []
        for tool_call in tools_called:
            formatted_call = f"Tool: {tool_call.name}\nParameters: {json.dumps(tool_call.input_parameters, indent=2)}"
            if tool_call.description:
                formatted_call += f"\nDescription: {tool_call.description}"
            formatted_calls.append(formatted_call)
        output = "\n\n".join(formatted_calls)

    return OutputParameters(
        output=output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tools_called=tools_called,
    )


def extract_output_parameters_from_response(
    response: Response, input_parameters: InputParameters
) -> OutputParameters:
    output = response.output_text
    prompt_tokens = response.usage.input_tokens
    completion_tokens = response.usage.output_tokens

    # Extract Tool Calls
    tools_called = None
    openai_raw_output = response.output
    if openai_raw_output is not None:
        tools_called = []
        for tool_call in openai_raw_output:
            if tool_call.type != "function_call":
                continue
            tool_descriptions = input_parameters.tool_descriptions or {}
            tools_called.append(
                ToolCall(
                    name=tool_call.name,
                    input_parameters=json.loads(tool_call.arguments),
                    description=tool_descriptions.get(tool_call.name),
                )
            )

    return OutputParameters(
        output=output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tools_called=tools_called,
    )
