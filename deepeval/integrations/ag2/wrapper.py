import json
import logging
from functools import wraps

from autogen import ConversableAgent
from autogen.oai.client import OpenAIWrapper

from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.tracing.types import LlmSpan, ToolSpan, TraceSpanStatus

logger = logging.getLogger(__name__)

_ORIGINAL_METHODS = {}


def wrap_generate_reply():
    original = ConversableAgent.generate_reply
    _ORIGINAL_METHODS["generate_reply"] = original

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        agent_name = getattr(self, "name", "unknown_agent")

        with Observer(
            span_type="agent",
            func_name=agent_name,
            observe_kwargs={"name": agent_name},
        ) as observer:
            result = original(self, *args, **kwargs)
            observer.result = str(result) if result else None

        return result

    ConversableAgent.generate_reply = wrapper


def wrap_a_generate_reply():
    original = ConversableAgent.a_generate_reply
    _ORIGINAL_METHODS["a_generate_reply"] = original

    @wraps(original)
    async def wrapper(self, *args, **kwargs):
        agent_name = getattr(self, "name", "unknown_agent")

        with Observer(
            span_type="agent",
            func_name=agent_name,
            observe_kwargs={"name": agent_name},
        ) as observer:
            result = await original(self, *args, **kwargs)
            observer.result = str(result) if result else None

        return result

    ConversableAgent.a_generate_reply = wrapper


def wrap_execute_function():
    original = ConversableAgent.execute_function
    _ORIGINAL_METHODS["execute_function"] = original

    @wraps(original)
    def wrapper(self, func_call, call_id=None, verbose=False):
        func_name = func_call.get("name", "unknown_tool")
        raw_args = func_call.get("arguments", "{}")
        try:
            tool_input = (
                json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            )
        except (json.JSONDecodeError, TypeError):
            tool_input = {"raw": str(raw_args)}

        exec_failed = False

        with Observer(
            span_type="tool",
            func_name=func_name,
            observe_kwargs={"name": func_name},
            function_kwargs=tool_input,
        ) as observer:
            is_exec_success, result_dict = original(
                self, func_call, call_id=call_id, verbose=verbose
            )

            content = result_dict.get("content", "")
            observer.result = content
            exec_failed = not is_exec_success

            span = trace_manager.get_span_by_uuid(observer.uuid)
            if span and isinstance(span, ToolSpan):
                span.input = tool_input
                span.output = str(content)[:2000] if content else ""

            def _update_on_failure(s):
                if exec_failed:
                    s.status = TraceSpanStatus.ERRORED

            observer.update_span_properties = _update_on_failure

        return is_exec_success, result_dict

    ConversableAgent.execute_function = wrapper


def wrap_a_execute_function():
    original = ConversableAgent.a_execute_function
    _ORIGINAL_METHODS["a_execute_function"] = original

    @wraps(original)
    async def wrapper(self, func_call, call_id=None, verbose=False):
        func_name = func_call.get("name", "unknown_tool")
        raw_args = func_call.get("arguments", "{}")
        try:
            tool_input = (
                json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            )
        except (json.JSONDecodeError, TypeError):
            tool_input = {"raw": str(raw_args)}

        exec_failed = False

        with Observer(
            span_type="tool",
            func_name=func_name,
            observe_kwargs={"name": func_name},
            function_kwargs=tool_input,
        ) as observer:
            is_exec_success, result_dict = await original(
                self, func_call, call_id=call_id, verbose=verbose
            )

            content = result_dict.get("content", "")
            observer.result = content
            exec_failed = not is_exec_success

            span = trace_manager.get_span_by_uuid(observer.uuid)
            if span and isinstance(span, ToolSpan):
                span.input = tool_input
                span.output = str(content)[:2000] if content else ""

            def _update_on_failure(s):
                if exec_failed:
                    s.status = TraceSpanStatus.ERRORED

            observer.update_span_properties = _update_on_failure

        return is_exec_success, result_dict

    ConversableAgent.a_execute_function = wrapper


def wrap_openai_wrapper_create():
    original = OpenAIWrapper.create
    _ORIGINAL_METHODS["openai_wrapper_create"] = original

    @wraps(original)
    def wrapper(self, **config):
        messages = config.get("messages", None)
        model = None
        config_list = getattr(self, "_config_list", None)
        if config_list:
            first = config_list[0]
            if isinstance(first, dict):
                model = first.get("model", None)
            else:
                model = getattr(first, "model", None)

        with Observer(
            span_type="llm",
            func_name="llm_call",
            observe_kwargs={"model": model},
        ) as observer:
            response = original(self, **config)
            observer.result = None

            span = trace_manager.get_span_by_uuid(observer.uuid)
            if span and isinstance(span, LlmSpan):
                if messages:
                    span.input = messages

                # Extract model name from response
                response_model = getattr(response, "model", None)
                if response_model:
                    span.model = response_model

                # Extract output text
                try:
                    extracted = self.extract_text_or_completion_object(response)
                    if extracted:
                        output = extracted[0]
                        if hasattr(output, "model_dump"):
                            span.output = output.model_dump()
                        else:
                            span.output = str(output)
                except Exception:
                    pass

                # Extract token usage
                usage = getattr(response, "usage", None)
                if usage:
                    span.input_token_count = getattr(
                        usage, "prompt_tokens", None
                    )
                    span.output_token_count = getattr(
                        usage, "completion_tokens", None
                    )

        return response

    OpenAIWrapper.create = wrapper


def unwrap_all():
    """Restore all original methods."""
    if "generate_reply" in _ORIGINAL_METHODS:
        ConversableAgent.generate_reply = _ORIGINAL_METHODS["generate_reply"]
    if "a_generate_reply" in _ORIGINAL_METHODS:
        ConversableAgent.a_generate_reply = _ORIGINAL_METHODS[
            "a_generate_reply"
        ]
    if "execute_function" in _ORIGINAL_METHODS:
        ConversableAgent.execute_function = _ORIGINAL_METHODS[
            "execute_function"
        ]
    if "a_execute_function" in _ORIGINAL_METHODS:
        ConversableAgent.a_execute_function = _ORIGINAL_METHODS[
            "a_execute_function"
        ]
    if "openai_wrapper_create" in _ORIGINAL_METHODS:
        OpenAIWrapper.create = _ORIGINAL_METHODS["openai_wrapper_create"]

    _ORIGINAL_METHODS.clear()
