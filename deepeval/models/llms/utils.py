from typing import Dict
import re
import json
import asyncio

from deepeval.errors import DeepEvalError

MULTIMODAL_MODELS = ["GPTModel", "AzureModel", "GeminiModel", "OllamaModel"]


def trim_and_load_json(
    input_string: str,
) -> Dict:
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


def safe_asyncio_run(coro):
    """
    Run an async coroutine safely, regardless of whether an event loop is
    already running (e.g. inside Jupyter notebooks, FastAPI, pytest-asyncio).

    Strategy:
    - If a loop IS running: apply nest_asyncio (idempotent) so the coroutine
      can be driven on the existing loop without a nested-loop RuntimeError.
    - If no loop is running: use the standard asyncio.run() path.
    """
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None:
        # A loop is already active — patch it with nest_asyncio so we can
        # call run_until_complete from within it, then schedule the coroutine.
        import nest_asyncio

        nest_asyncio.apply(running_loop)
        return running_loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)
