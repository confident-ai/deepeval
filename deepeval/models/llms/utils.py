from typing import Dict
import re
import json
import asyncio

from deepeval.errors import DeepEvalError

MULTIMODAL_MODELS = ["GPTModel", "AzureModel", "GeminiModel", "OllamaModel"]


def trim_and_load_json(
    input_string: str,
) -> Dict:
    # Extract the LAST balanced JSON object (the judge's verdict), not the span
    # from the first { to the last }. This prevents verdict injection: a
    # model-under-test that embeds {"verdict":"yes"} in its output can hijack
    # the verdict when the judge references that output in its reasoning.
    jsonStr = _extract_last_json_object(input_string)
    if not jsonStr:
        # Fallback: original find/rfind for incomplete JSON (missing closing brace).
        start = input_string.find("{")
        end = input_string.rfind("}") + 1
        if end == 0 and start != -1:
            input_string = input_string + "}"
            end = len(input_string)
        jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        raise DeepEvalError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def _extract_last_json_object(text: str) -> str:
    """Find the last balanced JSON object by scanning backwards from the final
    } and tracking brace depth. Ignores JSON from model output that the judge
    referenced in its reasoning."""
    last_close = text.rfind("}")
    if last_close == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(last_close, -1, -1):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            backslashes = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i : last_close + 1]
    return ""


def safe_asyncio_run(coro):
    """
    Run an async coroutine safely.
    Falls back to run_until_complete if already in a running event loop.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.ensure_future(coro)
                return loop.run_until_complete(future)
            else:
                return loop.run_until_complete(coro)
        except Exception:
            raise
    except Exception:
        raise
