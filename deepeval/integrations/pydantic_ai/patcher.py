import functools
from pydantic_ai.agent import AgentRunResult
from deepeval.tracing.types import AgentSpan
from deepeval.tracing.tracing import Observer
from typing import List
from deepeval.test_case.llm_test_case import ToolCall
try:
    from pydantic_ai.agent import Agent
    pydantic_ai_installed = True
except:
    pydantic_ai_installed = True

def _patch_agent_run():
    original_run = Agent.run

    @functools.wraps(original_run)
    async def wrapper(*args, **kwargs):
        with Observer(
            span_type="agent",
            func_name="Agent",
            function_kwargs={"input": args[1]},
        ) as observer:
            result = await original_run(*args, **kwargs)
            observer.update_span_properties = (
                lambda agent_span: update_agent_span_properties(
                    agent_span, result
                )
            )
            observer.result = result
        return result

    Agent.run = wrapper

def patch_all():
    _patch_agent_run()

def update_agent_span_properties(agent_span: AgentSpan, result: AgentRunResult):
    agent_span.tools_called = _extract_tools_called(result)

def _extract_tools_called(result: AgentRunResult) -> List[ToolCall]:
    tool_calls = []
    
    # Access the message history from the _state
    message_history = result._state.message_history
    
    # Scan through all messages in the history
    for message in message_history:
        # Check if this is a ModelResponse (kind="response")
        if hasattr(message, 'kind') and message.kind == "response":
            # For ModelResponse messages, check each part
            if hasattr(message, 'parts'):
                for part in message.parts:
                    # Look for parts with part_kind="tool-call"
                    if hasattr(part, 'part_kind') and part.part_kind == "tool-call":
                        # Extract tool name and args from the ToolCallPart
                        tool_name = part.tool_name
                        input_parameters = part.args_as_dict() if hasattr(part, 'args_as_dict') else None
                        
                        # Create and append ToolCall object
                        tool_call = ToolCall(
                            name=tool_name,
                            input_parameters=input_parameters
                        )
                        tool_calls.append(tool_call)
    
    print(tool_calls)
    return tool_calls
    