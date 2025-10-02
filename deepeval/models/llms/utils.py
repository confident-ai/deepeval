from typing import Dict
import re
import json
import asyncio


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
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


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
        except Exception as inner_e:
            raise
    except Exception as e:
        raise
