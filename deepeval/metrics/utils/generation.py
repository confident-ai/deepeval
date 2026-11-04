import json
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
    BaseArenaMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.utils import EvaluationCost
from deepeval.test_case import Audio

from .models import is_native_model


_ESCAPE_PAIR = re.compile(r"\\(.?)", re.DOTALL)
_VALID_ESCAPE_CHARS = set('"\\/bfnrtu')


def _repair_escapes(text: str) -> str:
    def fix(m):
        nxt = m.group(1)
        return (
            m.group(0) if nxt and nxt in _VALID_ESCAPE_CHARS else "\\\\" + nxt
        )

    return _ESCAPE_PAIR.sub(fix, text)


_TRAILING_COMMA = re.compile(r",\s*([\]}])")

EMPTY_OUTPUT_ERROR = (
    "Evaluation LLM returned an empty response, so there was no JSON to parse. "
    "With reasoning models (e.g. the gpt-5 family, o-series, gpt-oss) this usually "
    "means the completion-token limit was used up by hidden reasoning before any "
    "output was written. Raise the max completion tokens or lower reasoning effort "
    "on your evaluation model."
)
TRUNCATED_OUTPUT_ERROR = (
    "Evaluation LLM output appears truncated: it starts a JSON object but never "
    "closes it. This usually means the completion-token limit was reached "
    "(reasoning models spend part of that budget on hidden reasoning). "
    "Raise the max completion tokens on your evaluation model."
)
INVALID_JSON_ERROR = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."


def _fail(message: str, metric: Optional[Any]) -> None:
    if metric is not None:
        metric.error = message
    raise ValueError(message)


def trimAndLoadJson(
    input_string: Optional[str], metric: Optional[Any] = None
) -> Any:
    if input_string is None or not input_string.strip():
        _fail(EMPTY_OUTPUT_ERROR, metric)

    start = input_string.find("{")
    end = input_string.rfind("}") + 1
    if start == -1:
        # no object at all — let json.loads judge the whole string
        # (it may be a bare list, or a plain-text non-JSON reply)
        start, end = 0, len(input_string)
    elif end <= start:
        _fail(TRUNCATED_OUTPUT_ERROR, metric)

    json_str = _TRAILING_COMMA.sub(r"\1", input_string[start:end])

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Repair illegal backslash escapes (#2280's "Invalid \escape") and retry once.
    repaired = _repair_escapes(json_str)
    if repaired != json_str:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    _fail(INVALID_JSON_ERROR, metric)


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


def audio_kwargs(audio: Optional[List[Audio]]) -> Dict[str, Any]:
    # Only pass `audio` when there is some, so custom models whose generate()
    # predates the parameter keep working.
    return {"audio": audio} if audio else {}


def resolve_judge(
    metric: Union[BaseMetric, BaseArenaMetric, BaseConversationalMetric],
    model: Optional[DeepEvalBaseLLM],
) -> Tuple[DeepEvalBaseLLM, bool]:
    # A metric can score some calls with a model other than its main one (e.g.
    # an audio model), so the judge and its nativeness come from the call.
    if model is None or model is metric.model:
        return metric.model, metric.using_native_model
    return model, is_native_model(model)


def generate_with_schema_and_extract(
    metric: Union[BaseMetric, BaseArenaMetric, BaseConversationalMetric],
    prompt: Any,
    schema_cls: Type[SchemaType],
    *,
    extract_schema: Callable[[SchemaType], ReturnType],
    extract_json: Callable[[Dict[str, Any]], ReturnType],
    audio: Optional[List[Audio]] = None,
    model: Optional[DeepEvalBaseLLM] = None,
) -> ReturnType:
    """
    Synchronous wrapper:
    - calls model.generate_with_schema(...)
    - accrues cost if applicable
    - if schema instance -> extract_schema
      else parse JSON -> extract_json
    """
    judge, native = resolve_judge(metric, model)
    if native:
        result, cost = judge.generate_with_schema(
            prompt, schema=schema_cls, **audio_kwargs(audio)
        )
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
    else:
        result = judge.generate_with_schema(
            prompt, schema=schema_cls, **audio_kwargs(audio)
        )
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
    audio: Optional[List[Audio]] = None,
    model: Optional[DeepEvalBaseLLM] = None,
) -> ReturnType:
    judge, native = resolve_judge(metric, model)
    if native:
        result, cost = await judge.a_generate_with_schema(
            prompt, schema=schema_cls, **audio_kwargs(audio)
        )
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
    else:
        result = await judge.a_generate_with_schema(
            prompt, schema=schema_cls, **audio_kwargs(audio)
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
