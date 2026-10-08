import json
import re
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
    BaseArenaMetric,
)
from deepeval.models.utils import EvaluationCost


def trimAndLoadJson(
    input_string: Optional[str],
    metric: Optional[BaseMetric] = None,
) -> Any:
    if input_string is None:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        if metric is not None:
            metric.error = error_str
        raise ValueError(error_str)

    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        # Some models emit a trailing comma before a closing ] or }. Strip it
        # and retry, but only after a direct parse fails, so valid JSON string
        # values containing ", ]" or ", }" are never corrupted.
        try:
            return json.loads(re.sub(r",\s*([\]}])", r"\1", jsonStr))
        except json.JSONDecodeError:
            error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
            if metric is not None:
                metric.error = error_str
            raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


SchemaType = TypeVar("SchemaType")
ReturnType = TypeVar("ReturnType")


def accrue_token_usage(
    metric: Union[BaseMetric, BaseArenaMetric, BaseConversationalMetric],
    cost: Optional[float],
) -> None:
    """Accrue the input/output token counts that produced ``cost`` onto the
    metric.

    Native models return their cost as an ``EvaluationCost`` (a ``float``
    subclass carrying ``input_tokens``/``output_tokens``). Costs from providers
    that aren't wrapped yet — or ``None`` when pricing is unknown — are plain
    floats with no token data, so tokens are only accrued when ``cost`` actually
    carries them. Call this right after ``metric._accrue_cost(cost)`` so token
    usage tracks cost exactly.
    """
    if isinstance(cost, EvaluationCost):
        metric._accrue_tokens(cost.input_tokens, cost.output_tokens)


def generate_with_schema_and_extract(
    metric: Union[BaseMetric, BaseArenaMetric, BaseConversationalMetric],
    prompt: Any,
    schema_cls: Type[SchemaType],
    *,
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
) -> ReturnType:
    """
    Synchronous wrapper:
    - calls model.generate_with_schema(...)
    - accrues cost if applicable
    - if schema instance -> extract_schema
      else parse JSON -> extract_json
    """
    if metric.using_native_model:
        result, cost = metric.model.generate_with_schema(
            prompt, schema=schema_cls
        )
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
    else:
        result = metric.model.generate_with_schema(prompt, schema=schema_cls)
    if isinstance(result, schema_cls):
        return extract_schema(result)
    data = trimAndLoadJson(result, metric)
    return extract_json(data)


async def a_generate_with_schema_and_extract(
    metric: Union[BaseMetric, BaseArenaMetric, BaseConversationalMetric],
    prompt: Any,
    schema_cls: Type[SchemaType],
    *,
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
) -> ReturnType:
    if metric.using_native_model:
        result, cost = await metric.model.a_generate_with_schema(
            prompt, schema=schema_cls
        )
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
    else:
        result = await metric.model.a_generate_with_schema(
            prompt, schema=schema_cls
        )

    # Handle models that return (result, cost) tuple even when not native
    if isinstance(result, tuple) and len(result) == 2:
        actual_result, cost = result
        if hasattr(metric, "_accrue_cost"):
            metric._accrue_cost(cost)
            accrue_token_usage(metric, cost)
        result = actual_result

    if isinstance(result, schema_cls):
        return extract_schema(result)

    data = trimAndLoadJson(result, metric)
    return extract_json(data)
