from pydantic import BaseModel
from .schema import SyntheticData, SyntheticDataList
import os
import time
import asyncio
import logging
from contextvars import ContextVar
from contextlib import contextmanager
from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.models import DeepEvalBaseLLM

def _max_retries() -> int:
    max_retries = os.getenv("DEEPTEAM_MAX_RETRIES")
    if max_retries is None:
        return 3
    try:
        return max(1, int(max_retries))
    except ValueError:
        return 3


def add_cost(*costs):
    """Combine simulation/evaluation costs where ``None`` means 'no cost
    reported' (non-native models). Returns ``None`` only when every input is
    ``None``; otherwise treats ``None`` as ``0``.
    """
    if all(c is None for c in costs):
        return None
    return sum(c or 0 for c in costs)


# Context-local simulation-cost accumulator. When a ``cost_accumulator()``
# scope is active, every ``generate``/``a_generate`` call records its cost into
# it. This lets callers (e.g. attack enhancement) capture the cost of an entire
# nested operation without threading cost through every method. ContextVars are
# concurrency-safe: each asyncio task gets its own copy, and child tasks created
# inside a scope share the same accumulator object.
_cost_acc: ContextVar = ContextVar("deepteam_simulation_cost_acc", default=None)


def _record_cost(cost):
    """Add ``cost`` to the active accumulator, if any is in scope."""
    acc = _cost_acc.get()
    if acc is not None:
        acc[0] = add_cost(acc[0], cost)


@contextmanager
def cost_accumulator():
    """Open a context-local accumulator that captures the simulation cost of all
    ``generate``/``a_generate`` calls made within the scope.

    Yields a single-element list whose ``[0]`` holds the running total (``None``
    until a native-model cost is recorded).
    """
    acc = [None]
    token = _cost_acc.set(acc)
    try:
        yield acc
    finally:
        _cost_acc.reset(token)


def current_simulation_cost():
    acc = _cost_acc.get()
    return acc[0] if acc is not None else None


def generate_with_cost(
    prompt: str,
    schema: BaseModel,
    model: DeepEvalBaseLLM = None,
):
    """
    Generate schema using the provided model with retry logic, returning cost.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        A tuple ``(res, cost)`` where ``res`` is the validated schema object and
        ``cost`` is the model-reported cost for native models, or ``None`` for
        non-native models that do not report cost.
    """
    _, using_native_model = initialize_model(model=model)
    last_error = None
    retry_prompt = prompt
    max_retries = _max_retries()

    for attempt in range(max_retries):
        try:
            if using_native_model:
                res, cost = model.generate(prompt=retry_prompt, schema=schema)
                if res is None:
                    raise ValueError(
                        "Model returned None. This could be because of your model's guardrails, please use a different model."
                    )
                return res, cost
            else:
                try:
                    res = model.generate(prompt=retry_prompt, schema=schema)
                    if res is None:
                        raise ValueError(
                            "Model returned None. This could be because of your model's guardrails, please use a different model."
                        )

                    if isinstance(res, str):
                        data = trimAndLoadJson(res)
                        return schema(**data), None
                    else:
                        return res, None
                except TypeError:
                    res = model.generate(retry_prompt)
                    if res is None:
                        raise ValueError(
                            "Model returned None. This could be because of your model's guardrails, please use a different model."
                        )

                    data = trimAndLoadJson(res)
                    if schema == SyntheticDataList:
                        data_list = [
                            SyntheticData(**item) for item in data["data"]
                        ]
                        return SyntheticDataList(data=data_list), None
                    else:
                        return schema(**data), None

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                from deepteam.attacks.single_turn.escalation_constants import (
                    append_critic_feedback,
                    random_escalation_suffix,
                )

                escalated_prompt = (
                    f"{random_escalation_suffix(attempt)} \n\n {prompt}"
                )
                retry_prompt = append_critic_feedback(escalated_prompt, str(e))
                sleep_time = 2**attempt
                logging.warning(
                    f"Generation failed on attempt {attempt + 1}. Retrying in {sleep_time}s... Error: {e}"
                )
                time.sleep(sleep_time)

    raise RuntimeError(
        f"Failed to generate after {max_retries} attempts. Last error: {last_error}"
    )


def generate(
    prompt: str,
    schema: BaseModel,
    model: DeepEvalBaseLLM = None,
) -> BaseModel:
    """Backward-compatible wrapper around :func:`generate_with_cost` that
    returns only the validated schema object. Records cost into the active
    :func:`cost_accumulator` scope, if any."""
    res, cost = generate_with_cost(prompt, schema, model)
    _record_cost(cost)
    return res


async def a_generate_with_cost(
    prompt: str,
    schema: BaseModel,
    model: "DeepEvalBaseLLM" = None,
):
    """
    Asynchronously generate schema using the provided model with retry logic,
    returning cost.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        A tuple ``(res, cost)`` where ``res`` is the validated schema object and
        ``cost`` is the model-reported cost for native models, or ``None`` for
        non-native models that do not report cost.
    """
    _, using_native_model = initialize_model(model=model)
    last_error = None
    retry_prompt = prompt
    max_retries = _max_retries()

    for attempt in range(max_retries):
        try:
            if using_native_model:
                res, cost = await model.a_generate(
                    prompt=retry_prompt, schema=schema
                )
                if res is None:
                    raise ValueError(
                        "Model returned None. This could be because of your model's guardrails, please use a different model."
                    )
                return res, cost
            else:
                try:
                    res = await model.a_generate(
                        prompt=retry_prompt, schema=schema
                    )
                    if res is None:
                        raise ValueError(
                            "Model returned None. This could be because of your model's guardrails, please use a different model."
                        )

                    if isinstance(res, str):
                        data = trimAndLoadJson(res)
                        return schema(**data), None
                    else:
                        return res, None
                except TypeError:
                    res = await model.a_generate(retry_prompt)
                    if res is None:
                        raise ValueError(
                            "Model returned None. This could be because of your model's guardrails, please use a different model."
                        )

                    data = trimAndLoadJson(res)
                    if schema == SyntheticDataList:
                        data_list = [
                            SyntheticData(**item) for item in data["data"]
                        ]
                        return SyntheticDataList(data=data_list), None
                    else:
                        return schema(**data), None

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                from deepteam.attacks.single_turn.escalation_constants import (
                    append_critic_feedback,
                    random_escalation_suffix,
                )

                escalated_prompt = (
                    f"{random_escalation_suffix(attempt)} \n\n {prompt}"
                )
                retry_prompt = append_critic_feedback(escalated_prompt, str(e))
                sleep_time = 2**attempt
                logging.warning(
                    f"Async generation failed on attempt {attempt + 1}. Retrying in {sleep_time}s... Error: {e}"
                )
                await asyncio.sleep(sleep_time)

    raise RuntimeError(
        f"Failed to async generate after {max_retries} attempts. Last error: {last_error}"
    )


async def a_generate(
    prompt: str,
    schema: BaseModel,
    model: "DeepEvalBaseLLM" = None,
) -> BaseModel:
    """Backward-compatible wrapper around :func:`a_generate_with_cost` that
    returns only the validated schema object. Records cost into the active
    :func:`cost_accumulator` scope, if any."""
    res, cost = await a_generate_with_cost(prompt, schema, model)
    _record_cost(cost)
    return res
