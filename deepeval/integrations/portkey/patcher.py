import json
import functools
from typing import Optional, List
from deepeval.tracing.context import current_span_context
from deepeval.tracing.tracing import Observer
from deepeval.tracing.utils import make_json_serializable
from deepeval.metrics import BaseMetric
from deepeval.tracing.types import LlmOutput, LlmToolCall
try:
    from portkey_ai import Portkey
    from portkey_ai.api_resources.apis.chat_complete import Completions, ChatCompletions
    is_portkey_installed = True
except Exception:
    is_portkey_installed = False

def is_portkey_available():
    if not is_portkey_installed:
        raise ImportError("portkey-ai is not available. Please install it with `pip install portkey-ai`.")
    return True

def _patch_portkey_init():
    original_init = Portkey.__init__

    @functools.wraps(original_init)
    def new_init(*args, metric_collection: Optional[str] = None, metrics: Optional[List[BaseMetric]] = None, **kwargs):
        result = original_init(*args, **kwargs)
        _patch_portkey_chat_completions(args[0].chat.completions, metrics, metric_collection)
        return result

    Portkey.__init__ = new_init

def _patch_portkey_chat_completions(completions: Completions, metrics=None, metric_collection=None):
    if getattr(completions, "_deepeval_patched", False):
        return

    original_create = completions.create  # capture bound original
    
    @functools.wraps(original_create)
    def new_create(*args, **kwargs):
        with Observer(
            span_type="llm",
            metrics=metrics,
            metric_collection=metric_collection,
            observe_kwargs={"model": kwargs.get("model", "unknown")},
            func_name="LLM",
        ) as observer:
            result = original_create(*args, **kwargs)
            observer.result = extract_llm_output_from_chat_completion_messages(result)
            current_span_context.get().input = kwargs.get("messages")
        return result

    completions.create = new_create
    setattr(completions, "_deepeval_patched", True)


def instrument():
    is_portkey_available()
    _patch_portkey_init()

def extract_llm_output_from_chat_completion_messages(result: ChatCompletions) -> List[LlmOutput]:
    try:
        # Local imports to avoid changing global import section/line numbers

        choices = None
        if hasattr(result, "choices"):
            choices = result.choices
        elif isinstance(result, dict):
            choices = result.get("choices")

        outputs = []
        if isinstance(choices, list):
            for c in choices:
                message = None
                if hasattr(c, "message"):
                    message = c.message
                elif isinstance(c, dict):
                    message = c.get("message")
                if message is None:
                    continue

                # role
                role = None
                if hasattr(message, "role"):
                    role = message.role
                elif isinstance(message, dict):
                    role = message.get("role")
                role = "AI" if (role or "").lower() == "assistant" else (role or "AI")

                # content
                raw_content = None
                if hasattr(message, "content"):
                    raw_content = message.content
                elif isinstance(message, dict):
                    raw_content = message.get("content")

                content_str = ""
                if isinstance(raw_content, str):
                    content_str = raw_content
                elif isinstance(raw_content, list):
                    parts = []
                    for part in raw_content:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            text = part.get("text") or part.get("content") or part.get("value")
                            if text is not None:
                                parts.append(str(text))
                            elif part.get("type") == "text" and "text" in part:
                                parts.append(str(part["text"]))
                    content_str = "".join(parts)
                elif raw_content is not None:
                    content_str = str(raw_content)

                # tool calls (tool_calls or function_call)
                tool_calls = []
                tcs = None
                if hasattr(message, "tool_calls"):
                    tcs = message.tool_calls
                elif isinstance(message, dict):
                    tcs = message.get("tool_calls")

                if isinstance(tcs, list):
                    for tc in tcs:
                        func = None
                        tc_id = None
                        if hasattr(tc, "function"):
                            func = tc.function
                        elif isinstance(tc, dict):
                            func = tc.get("function")
                        if hasattr(tc, "id"):
                            tc_id = tc.id
                        elif isinstance(tc, dict):
                            tc_id = tc.get("id")

                        name = None
                        arguments = None
                        if func is not None:
                            if hasattr(func, "name"):
                                name = func.name
                            elif isinstance(func, dict):
                                name = func.get("name")
                            if hasattr(func, "arguments"):
                                arguments = func.arguments
                            elif isinstance(func, dict):
                                arguments = func.get("arguments")

                        args_obj = {}
                        if isinstance(arguments, str):
                            try:
                                args_obj = json.loads(arguments)
                            except Exception:
                                args_obj = {"arguments": arguments}
                        elif isinstance(arguments, dict):
                            args_obj = arguments
                        elif arguments is not None:
                            args_obj = {"arguments": arguments}

                        if name:
                            tool_calls.append(LlmToolCall(name=name, args=args_obj, id=tc_id))
                else:
                    # fallback to single function_call
                    fc = None
                    if hasattr(message, "function_call"):
                        fc = message.function_call
                    elif isinstance(message, dict):
                        fc = message.get("function_call")
                    if fc:
                        name = None
                        arguments = None
                        if hasattr(fc, "name"):
                            name = fc.name
                        elif isinstance(fc, dict):
                            name = fc.get("name")
                        if hasattr(fc, "arguments"):
                            arguments = fc.arguments
                        elif isinstance(fc, dict):
                            arguments = fc.get("arguments")

                        args_obj = {}
                        if isinstance(arguments, str):
                            try:
                                args_obj = json.loads(arguments)
                            except Exception:
                                args_obj = {"arguments": arguments}
                        elif isinstance(arguments, dict):
                            args_obj = arguments
                        elif arguments is not None:
                            args_obj = {"arguments": arguments}

                        if name:
                            tool_calls.append(LlmToolCall(name=name, args=args_obj))

                outputs.append(
                    LlmOutput(
                        role=role,
                        content=content_str or "",
                        tool_calls=tool_calls or None,
                    )
                )

        return outputs if outputs else make_json_serializable(result)
    except Exception:
        return make_json_serializable(result)