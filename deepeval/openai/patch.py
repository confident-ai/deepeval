from typing import Dict, Optional, List
from openai.types import Completion
from pydantic import BaseModel
from functools import wraps
import json
import uuid
from deepeval.test_case import LLMTestCase

from deepeval.tracing.context import update_current_span, current_span_context, update_current_trace
from deepeval.tracing.attributes import LlmAttributes, ToolAttributes
from deepeval.tracing.types import AgentSpan, BaseSpan, LlmSpan, ToolSpan, RetrieverSpan, TraceSpanStatus
from deepeval.test_run import auto_log_hyperparameters
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing import trace_manager
# from deepeval.prompt.api import PromptMessage
# from deepeval.prompt.prompt import Prompt

class InputParameters(BaseModel):
    model: Optional[str] = None
    input: Optional[str] = None
    instructions: Optional[str] = None
    messages: Optional[List[Dict]] = None
    tools: Optional[List[Dict]]=None

    
class OutputParameters(BaseModel):
    output: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tools_called: Optional[List[ToolCall]] = None


def patch_openai(openai_module):
    if getattr(openai_module, "_deepeval_patched", False): return
    openai_module._deepeval_patched = True

    for openai_client_name in ("OpenAI", "AsyncOpenAI"):
        openai_client = getattr(openai_module, openai_client_name, None)
        if not (openai_client and isinstance(openai_client, type)):
            continue
        original_init_fn = openai_client.__init__
        setattr(openai_client, "__init__", create_openai_client_init_wrapper(original_init_fn))

def create_openai_client_init_wrapper(original_init_fn):
    @wraps(original_init_fn)
    def new_init_fn(self, *args, **kwargs):
        original_init_fn(self, *args, **kwargs)
        wrap_openai_client_methods(self)
    return new_init_fn
            

##############################################
# Wrap methods in OpenAI Client
##############################################

def wrap_openai_client_methods(client):
    # 1) wrap client.responses.create
    if hasattr(client, "responses"):
        responses_class = client.responses
        original_method = getattr(responses_class, "create", None)
        if callable(original_method):
            setattr(
                responses_class,
                "create",
                create_openai_method_wrapper(original_method)
            )

    # 2) wrap client.chat.completions.create
    chat_class = getattr(client, "chat", None)
    if chat_class and hasattr(chat_class, "completions"):
        completions_class = chat_class.completions
        original_method = getattr(completions_class, "create", None)
        if callable(original_method):
            setattr(
                completions_class,
                "create",
                create_openai_method_wrapper(original_method)
            )

    # 3) wrap client.beta.chat.completions.parse
    beta_class = getattr(client, "beta", None)
    if beta_class and hasattr(beta_class, "chat"):
        chat_class = getattr(client, "chat", None)
        if chat_class and hasattr(chat_class, "completions"):
            completions_class = chat_class.completions
            original_method = getattr(completions_class, "parse", None)
            if callable(original_method):
                setattr(
                    completions_class,
                    "parse",
                    create_openai_method_wrapper(original_method)
                )

    
#####################################################
# Wrapper for Individual Method
#####################################################

def create_openai_method_wrapper(orig_fn):
    @wraps(orig_fn)
    def openai_method_wrapper(*args, **kwargs):
        input_parameters = extract_input_parameters(kwargs)
        response = orig_fn(*args, **kwargs)
        output_parameters = extract_output_parameters(response, input_parameters)
        if trace_manager.evaluating == True:
            log_hyperparameters(input_parameters)
        update_span_attributes(input_parameters, output_parameters)
        update_test_case(input_parameters, output_parameters)
        update_child_spans(output_parameters)
        return response
    return openai_method_wrapper


#####################################################
# Extracing Parameters from OpenAI
#####################################################

def extract_input_parameters(kwargs: Dict):
    model = kwargs.get('model')
    input = kwargs.get('input')
    instructions = kwargs.get('instructions')
    messages = kwargs.get('messages')
    tools = kwargs.get('tools')
    return InputParameters(
        model=model, input=input, 
        instructions=instructions, 
        messages=messages, 
        tools=tools
    ) 

def extract_output_parameters(
    response: Completion, 
    input_parameters: InputParameters
) -> OutputParameters:
    # Extract Output
    output = getattr(response, "output_text", None)
    if output is None:
        try:
            output = response.choices[0].message.content
        except (IndexError, AttributeError):
            pass
    # Extract Input/Output Tokens
    prompt_tokens = None
    completion_tokens = None
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    # Extract Tool Calls
    tools_called = None
    response_output = getattr(response, "output", None)
    if response_output is not None:
        tool_descriptions = (
            {tool["name"]: tool["description"] for tool in input_parameters.tools}
            if input_parameters.tools is not None
            else None
        )
        tools_called = []
        for tool_call in response_output:
            if tool_call.type != "function_call":
                continue
            tools_called.append(
                ToolCall(
                    name=tool_call.name,
                    input_parameters=json.loads(tool_call.arguments),
                    description=tool_descriptions[tool_call.name],
                )
            )
    else:
        try:
            response_output = response.choices[0].message.tool_calls
            if response_output is None: raise AttributeError
            tool_descriptions = (
                {tool["function"]["name"]: tool["function"]["description"] for tool in input_parameters.tools}
                if input_parameters.tools is not None
                else None
            )
            tools_called = []
            for tool_call in response_output:
                tools_called.append(
                    ToolCall(
                        name=tool_call.function.name,
                        input_parameters=json.loads(tool_call.function.arguments),
                        description=tool_descriptions[tool_call.function.name],
                    )
                )
        except (IndexError, AttributeError):
            pass
    return OutputParameters(
        output=output,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tools_called=tools_called
    )


#####################################################
# Update Span and Log Hyperparameters
#####################################################

def update_span_attributes(
    input_parameters: InputParameters,
    output_parameters: OutputParameters
):  
    # skip if span not LlmSpan
    # skip if span not LlmSpan or BaseSpan
    current_span = current_span_context.get()
    is_base_span = (
        not isinstance(current_span, AgentSpan) and 
        not isinstance(current_span, ToolSpan) and 
        not isinstance(current_span, RetrieverSpan) and
        not isinstance(current_span, LlmSpan)
    )
    if not isinstance(current_span, LlmSpan) and not is_base_span:
        return
    if is_base_span:
        new_span_instance = LlmSpan(**current_span.model_dump())
        current_span_context.set(new_span_instance)
        trace_manager.remove_span(current_span.uuid)
        trace_manager.remove_span_from_trace(current_span)
        trace_manager.add_span(new_span_instance)
        trace_manager.add_span_to_trace(new_span_instance)
        current_span = current_span_context.get()
    # update span attributes
    current_span.model = input_parameters.model or current_span.model
    update_current_span(
        attributes=LlmAttributes(
            input=input_parameters.input or input_parameters.messages or "NA",
            output=output_parameters.output or "NA",
            input_token_count=output_parameters.prompt_tokens,
            output_token_count=output_parameters.completion_tokens,
        )
    )

def update_test_case(
    input_parameters: InputParameters,
    output_parameters: OutputParameters,
    update_span: bool = True,
    update_trace: bool = True,
):  
    # skip if span not LlmSpan
    current_span = current_span_context.get()
    if not isinstance(current_span, LlmSpan):
        return
    # skip if input or output is None
    output = output_parameters.output
    input = input_parameters.input
    messages = input_parameters.messages
    if input is None and messages is not None and len(messages) > 0:
        user_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user" and content:
                user_messages.append(content)
        if len(user_messages) > 0:
            input = str(user_messages) if len(user_messages) > 1 else user_messages[0]
    if input is None or output is None:
        return
    # update current span
    if update_span:
        update_current_span(
            test_case=LLMTestCase(
                input=input,
                actual_output=output,
                tools_called=output_parameters.tools_called
            )
        )
    # update current trace
    if update_trace:
        update_current_trace(
            test_case=LLMTestCase(
                input=input,
                actual_output=output,
                tools_called=output_parameters.tools_called
            )
        )

def update_child_spans(
    output_parameters: OutputParameters
):
    # skip if span not LlmSpan
    current_span = current_span_context.get()
    if not isinstance(current_span, LlmSpan):
        return
    # add child ToolSpans to LlmSpan
    child_tool_spans = []
    if output_parameters.tools_called is None or len(output_parameters.tools_called) == 0: 
        return
    for tool_called in output_parameters.tools_called:
        tool_span = ToolSpan(
            **{
                "uuid": str(uuid.uuid4()),
                "trace_uuid": current_span.trace_uuid,
                "parent_uuid": current_span.uuid,
                "start_time": current_span.start_time,
                "end_time": current_span.start_time,
                "status": TraceSpanStatus.SUCCESS,
                "children": [],
                "name": tool_called.name,
                "input": tool_called.input_parameters,
                "output": None,
                "metrics": None,
                "attributes": ToolAttributes(
                    input=tool_called.input_parameters,
                    output=None
                ),
                "description": tool_called.description
            }
        )
        child_tool_spans.append(tool_span)
    current_span.children.extend(child_tool_spans)

def log_hyperparameters(input_parameters: InputParameters):
    # log model
    hyperparameters = {
        "model": input_parameters.model,
    }
    # log prompt (use instructions or messages)
    messages = input_parameters.messages
    instructions = input_parameters.instructions
    prompt = None
    if instructions is not None:
        prompt = instructions
    if messages is not None and len(messages) > 0:
        system_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system" and content:
                system_messages.append(content)
        if len(system_messages) > 0:
            prompt = str(system_messages) if len(system_messages) > 1 else system_messages[0]
    if prompt is not None:
        hyperparameters["system_prompt"] = prompt
    # auto log hyperparams
    auto_log_hyperparameters(hyperparameters)