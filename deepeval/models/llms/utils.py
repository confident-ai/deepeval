from typing import Dict
import re
import json
import asyncio

from deepeval.errors import DeepEvalError

MULTIMODAL_MODELS = ["GPTModel", "AzureModel", "GeminiModel", "OllamaModel"]


def _strip_think_tags(text: str) -> str:
    """Strip reasoning/thinking blocks emitted by models like DeepSeek-R1, QwQ,
    and Nemotron.  Handles both fully-paired ``<think>...</think>`` and the
    closing-tag-only variant where the chat template injects the opening tag so
    the model only emits ``</think>``."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if stripped == text:
        stripped = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    return stripped.strip()


def _extract_json(text: str) -> str:
    """Extract the outermost ``{...}`` substring from *text*."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if end == 0 and start != -1:
        text = text + "}"
        end = len(text)
    return text[start:end] if start != -1 and end != 0 else ""


def trim_and_load_json(
    input_string: str,
) -> Dict:
    jsonStr = _extract_json(input_string)

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        pass

    # Fallback 1: strip trailing commas before ] or }.
    try:
        return json.loads(re.sub(r",\s*([\]}])", r"\1", jsonStr))
    except json.JSONDecodeError:
        pass

    # Fallback 2: strip <think> blocks whose braces confused extraction,
    # then re-extract and parse.
    cleaned = _strip_think_tags(input_string)
    if cleaned != input_string:
        jsonStr2 = _extract_json(cleaned)
        try:
            return json.loads(jsonStr2)
        except json.JSONDecodeError:
            try:
                return json.loads(
                    re.sub(r",\s*([\]}])", r"\1", jsonStr2)
                )
            except json.JSONDecodeError:
                pass

    error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
    raise DeepEvalError(error_str)


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
