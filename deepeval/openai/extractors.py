from openai.types.chat import ChatCompletion, ParsedChatCompletion
from typing import Optional, Union, List, Dict
from openai.types.responses import Response
from pydantic import BaseModel
import json

from deepeval.test_case.llm_test_case import ToolCall


class InputParameters(BaseModel):
    model: Optional[str] = None
    input: Optional[str] = None
    instructions: Optional[str] = None
    messages: Optional[List[Dict]] = None
    tool_descriptions: Optional[Dict[str, str]] = None


class OutputParameters(BaseModel):
    output: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tools_called: Optional[List[ToolCall]] = None


def extract_input_parameters(
    is_completion: bool, kwargs: Dict
) -> InputParameters:
    if is_completion:
        return extract_input_parameters_from_completion(kwargs)
    else:
        return extract_input_parameters_from_response(kwargs)


def extract_input_parameters_from_completion(kwargs: Dict) -> InputParameters:
    model = kwargs.get("model")
    messages = kwargs.get("messages")
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
    input = ""
    user_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            user_messages.append(content)
    if len(user_messages) > 0:
        input = user_messages[0]

    return InputParameters(
        model=model,
        input=input,
        messages=messages,
        tools=tools,
        tool_descriptions=tool_descriptions_map,
    )


def extract_input_parameters_from_response(kwargs: Dict) -> InputParameters:
    model = kwargs.get("model")
    input = kwargs.get("input")
    instructions = kwargs.get("instructions")
    tools = kwargs.get("tools")
    tool_descriptions = (
        {tool["name"]: tool["description"] for tool in tools}
        if tools is not None
        else None
    )
    return InputParameters(
        model=model,
        input=input,
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
    output = str(completion.choices[0].message.content)
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    # Extract Tools Called
    tools_called = None
    openai_tool_calls = completion.choices[0].message.tool_calls
    if openai_tool_calls is not None:
        tools_called = []
        for tool_call in openai_tool_calls:
            tools_called.append(
                ToolCall(
                    name=tool_call.function.name,
                    input_parameters=json.loads(tool_call.function.arguments),
                    description=input_parameters.tool_descriptions.get(
                        tool_call.function.name
                    ),
                )
            )

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
            tools_called.append(
                ToolCall(
                    name=tool_call.name,
                    input_parameters=json.loads(tool_call.arguments),
                    description=input_parameters.tool_descriptions.get(
                        tool_call.name
                    ),
                )
            )

    return OutputParameters(
        output=output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tools_called=tools_called,
    )
