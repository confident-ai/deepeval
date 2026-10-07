import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from tenacity import RetryError

from deepeval.config.eval_mode import (
    EVAL_MODE_ENV_VAR,
    EvalMode,
    resolve_eval_mode,
)
from deepeval.errors import DeepEvalError
from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
)
from deepeval.models.system_one.limits import SystemOneContextLimitError
from deepeval.models.system_one.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneAnswers,
)

from .generation import (
    SchemaType,
    accrue_token_usage,
    trimAndLoadJson,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)

###############################################
# Decision points
###############################################
#
# The single place where a metric turns a prompt into a typed decision that is
# not a QAG verdict list: a DAG binary judgement, a DAG choice between options,
# or a G-Eval rubric score. Metrics build the prompt; these helpers own how the
# judge is asked. When the metric's eval mode uses System One the DAG
# judgements accept a `system_one` spec and ask a System One model instead of
# the LLM; the G-Eval score is LLM-only by design.

Metric = Union[BaseMetric, BaseConversationalMetric]

SYSTEM_ONE_YES_THRESHOLD = 0.5


###############################################
# System One (Jev) plumbing, shared with qag.py and classifiers/utils.py
###############################################
#
# Three eval modes (deepeval.config.eval_mode):
#
# - `llm`: `_system_one_active` is False, every helper takes its LLM branch.
# - `hybrid`: the helpers ask Jev. The LLM is already part of the chain, so
#   a Jev call that fails at runtime hands that one decision to the LLM
#   branch and records why on `metric.system_one_fallback_reason`.
# - `system_one`: the whole-chain runner in `system_one.py` asks Jev once per
#   measure and nothing falls back. A context overflow is re-raised telling
#   the user to switch back to `llm`; any other Jev error surfaces as-is.
#   Metrics with no whole-chain form run as `hybrid` (`effective_eval_mode`).


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def has_whole_metric_form(metric: Any) -> bool:
    """Whether the metric can run as one Jev request (it overrides
    ``_system_one_eval_spec``). Non-metrics such as classifiers, which
    handle `system_one` themselves, count as having one."""
    hook = getattr(type(metric), "_system_one_eval_spec", None)
    return hook is None or hook not in (
        BaseMetric._system_one_eval_spec,
        BaseConversationalMetric._system_one_eval_spec,
    )


def effective_eval_mode(metric: Any) -> EvalMode:
    """The mode the metric actually runs in. Metrics that take an
    ``eval_mode`` argument set ``metric.eval_mode`` at construction; anything
    else (a user's own metric subclass, a DAG node's parent) resolves from
    settings the same way ``initialize_system_one_model`` does.

    A metric with no whole-metric form (DAG, whose task nodes need the LLM,
    or a user's own subclass) cannot honour `system_one`, so it runs as
    `hybrid`: Jev at its decision points, the LLM covering a failed call."""
    mode = getattr(metric, "eval_mode", None)
    mode = mode if isinstance(mode, EvalMode) else resolve_eval_mode(mode)
    if mode is EvalMode.SYSTEM_ONE and not has_whole_metric_form(metric):
        return EvalMode.HYBRID
    return mode


def _system_one_active(metric: Any, spec: Optional[Any]) -> bool:
    if spec is None:
        return False
    mode = effective_eval_mode(metric)
    if not mode.uses_system_one:
        return False
    if getattr(metric, "_system_one_disabled", False):
        # `hybrid` only: an earlier Jev call in this measure failed in a way
        # that will not get better on retry (auth). Stay on the LLM path.
        return False
    if getattr(metric, "system_one_model", None) is None:
        raise DeepEvalError(
            f"{EVAL_MODE_ENV_VAR}={mode} routes decisions to a System One "
            f"model, but {type(metric).__name__} has none configured. Set "
            f"TYPESAFE_API_KEY or switch back with "
            f"`deepeval set-eval-mode {EvalMode.LLM}`."
        )
    return True


def reset_system_one_state(metric: Any) -> None:
    """Clear the per-measure System One bookkeeping. Called at the start of
    every measure (via the progress indicator every metric enters) so a
    reused metric instance never carries a previous test case's confidence
    or fallback reason into the next one."""
    metric.confidence = None
    metric.system_one_fallback_reason = None
    metric._system_one_outcomes = None
    metric._system_one_disabled = False


def _accrue(metric: Any, cost: Any) -> None:
    metric._accrue_cost(cost)
    accrue_token_usage(metric, cost)


def _record_confidence(metric: Any, answers: Any) -> Optional[float]:
    """Fold the least decisive answer of one Jev request into
    ``metric.confidence`` (minimum across the measure). ``answers`` is a
    ``SystemOneAnswers`` or one of its per-type maps."""
    if isinstance(answers, SystemOneAnswers):
        confidence = answers.min_confidence()
    else:
        values = [a.confidence for a in dict(answers).values()]
        confidence = min(values) if values else None
    if confidence is None:
        return None
    current = getattr(metric, "confidence", None)
    metric.confidence = (
        confidence if current is None else min(current, confidence)
    )
    return confidence


def _note_fallback(metric: Any, reason: str) -> None:
    previous = getattr(metric, "system_one_fallback_reason", None)
    metric.system_one_fallback_reason = (
        reason if not previous else f"{previous}; {reason}"
    )


def _unwrap(exc: BaseException) -> BaseException:
    if isinstance(exc, RetryError) and exc.last_attempt is not None:
        inner = exc.last_attempt.exception()
        if inner is not None:
            return inner
    return exc


def _is_permanent(exc: BaseException) -> bool:
    """Failures that will not get better on the next Jev call in this
    measure: no SDK, no key, rejected key."""
    if isinstance(exc, ImportError):
        return True
    try:
        from typesafe_sdk import (
            TypeSafeAuthenticationError,
            TypeSafePermissionDeniedError,
        )

        if isinstance(
            exc, (TypeSafeAuthenticationError, TypeSafePermissionDeniedError)
        ):
            return True
    except Exception:  # SDK optional
        pass
    text = str(exc).lower()
    return isinstance(exc, DeepEvalError) and (
        "api key" in text or "typesafe_sdk" in text or "typesafe-sdk" in text
    )


def _describe_failure(exc: BaseException) -> str:
    if isinstance(exc, SystemOneContextLimitError):
        return (
            f"context limit ({exc.estimated_tokens} est. tokens > "
            f"{exc.limit_tokens})"
        )
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return f"timeout ({exc})"
    return f"{type(exc).__name__}: {exc}"


def _system_one_error_types() -> Tuple[type, ...]:
    types: List[type] = [
        DeepEvalError,
        RetryError,
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        ImportError,
    ]
    try:
        from typesafe_sdk import TypeSafeError

        types.append(TypeSafeError)
    except Exception:  # SDK optional
        pass
    return tuple(types)


def context_limit_error(
    metric: Any, exc: SystemOneContextLimitError
) -> DeepEvalError:
    """The error a `system_one` metric raises when its test case does not
    fit in Jev's context: Jev cannot judge it, and the LLM can."""
    name = getattr(metric, "__name__", type(metric).__name__)
    return DeepEvalError(
        f"{name} could not run on System One: {exc} This test case is too "
        f"large for Jev to judge in one request. Switch this metric back to "
        f'the LLM with `eval_mode="{EvalMode.LLM}"` or '
        f"`deepeval set-eval-mode {EvalMode.LLM}`."
    )


def handle_system_one_failure(metric: Any, exc: BaseException) -> None:
    """Decide what a failed Jev call means for this metric.

    Under ``hybrid`` the failure is recorded on
    ``metric.system_one_fallback_reason`` and swallowed so the caller takes
    its LLM branch for that decision; permanent failures (auth) also keep the
    rest of the measure on the LLM. In every other mode it raises: a context
    overflow as the switch-back-to-``llm`` error, anything else unchanged.
    Programming errors (``KeyError`` ...) always propagate."""
    inner = _unwrap(exc)
    if not isinstance(inner, _system_one_error_types()):
        raise exc
    if effective_eval_mode(metric) is EvalMode.HYBRID:
        _note_fallback(metric, _describe_failure(inner))
        if _is_permanent(inner):
            metric._system_one_disabled = True
        return
    if isinstance(inner, SystemOneContextLimitError):
        raise context_limit_error(metric, inner) from inner
    raise exc


def _system_one_call(
    metric: Any, fn: Callable[..., Any], *args: Any
) -> Optional[Any]:
    """Run one System One model method. Returns its result, or ``None`` when
    the call failed and the metric (``hybrid`` only) takes its LLM branch."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - narrowed inside
        handle_system_one_failure(metric, e)
        return None


async def _a_system_one_call(
    metric: Any, fn: Callable[..., Any], *args: Any
) -> Optional[Any]:
    try:
        return await fn(*args)
    except Exception as e:  # noqa: BLE001 - narrowed inside
        handle_system_one_failure(metric, e)
        return None


@dataclass
class SystemOneBinarySpec:
    instructions: Any
    state: Dict[str, Any]


@dataclass
class SystemOneChoiceSpec:
    instructions: Any
    options: Sequence[str]
    state: Dict[str, Any]


###############################################
# Binary judgement
###############################################


def _binary_from_answer(
    schema_cls: Type[SchemaType], probability: float
) -> SchemaType:
    return schema_cls(
        verdict=probability >= SYSTEM_ONE_YES_THRESHOLD,
        reason=f"P(true)={probability:.2f}",
    )


def _binary_request(spec: SystemOneBinarySpec):
    return _jsonable(spec.state), {
        "verdict": NoulQuestion(instructions=spec.instructions)
    }


def generate_binary_judgement(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[SchemaType],
    system_one: Optional[SystemOneBinarySpec] = None,
) -> SchemaType:
    if _system_one_active(metric, system_one):
        result = _system_one_call(
            metric, metric.system_one_model.noul, *_binary_request(system_one)
        )
        if result is not None:
            answers, cost = result
            _accrue(metric, cost)
            _record_confidence(metric, answers)
            return _binary_from_answer(
                schema_cls, answers["verdict"].probability
            )

    return generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda s: s,
        extract_json=lambda data: schema_cls(**data),
    )


async def a_generate_binary_judgement(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[SchemaType],
    system_one: Optional[SystemOneBinarySpec] = None,
) -> SchemaType:
    if _system_one_active(metric, system_one):
        result = await _a_system_one_call(
            metric,
            metric.system_one_model.a_noul,
            *_binary_request(system_one),
        )
        if result is not None:
            answers, cost = result
            _accrue(metric, cost)
            _record_confidence(metric, answers)
            return _binary_from_answer(
                schema_cls, answers["verdict"].probability
            )

    return await a_generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda s: s,
        extract_json=lambda data: schema_cls(**data),
    )


###############################################
# Choice judgement
###############################################


def _choice_question(spec: SystemOneChoiceSpec) -> Dict[str, ChoiceQuestion]:
    return {
        "verdict": ChoiceQuestion(
            instructions=spec.instructions,
            options={option: None for option in spec.options},
        )
    }


def _choice_from_answer(
    schema_cls: Type[SchemaType], answer: Any
) -> SchemaType:
    return schema_cls(
        verdict=answer.choice,
        reason=(
            f"P={answer.probabilities.get(answer.choice, 0.0):.2f}, "
            f"confidence={answer.confidence:.2f}"
        ),
    )


def generate_choice_judgement(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[SchemaType],
    options: Sequence[str],
    system_one: Optional[SystemOneChoiceSpec] = None,
) -> SchemaType:
    if _system_one_active(metric, system_one):
        result = _system_one_call(
            metric,
            metric.system_one_model.choice,
            _jsonable(system_one.state),
            _choice_question(system_one),
        )
        if result is not None:
            answers, cost = result
            _accrue(metric, cost)
            _record_confidence(metric, answers)
            return _choice_from_answer(schema_cls, answers["verdict"])

    return generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda s: s,
        extract_json=lambda data: schema_cls(**data),
    )


async def a_generate_choice_judgement(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[SchemaType],
    options: Sequence[str],
    system_one: Optional[SystemOneChoiceSpec] = None,
) -> SchemaType:
    if _system_one_active(metric, system_one):
        result = await _a_system_one_call(
            metric,
            metric.system_one_model.a_choice,
            _jsonable(system_one.state),
            _choice_question(system_one),
        )
        if result is not None:
            answers, cost = result
            _accrue(metric, cost)
            _record_confidence(metric, answers)
            return _choice_from_answer(schema_cls, answers["verdict"])

    return await a_generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda s: s,
        extract_json=lambda data: schema_cls(**data),
    )


###############################################
# Single decisions
###############################################
#
# The decision point of a chain metric whose final LLM prompt returns one
# verdict or one score (Task Completion, Step Efficiency, ...). The metric
# still extracts with the LLM; under `hybrid` it asks Jev for the verdict or
# score first and only prompts the LLM when these return `None` (not
# deciding with Jev, or the Jev call failed and is handed to the LLM).


@dataclass
class SystemOneScoreSpec:
    instructions: Any
    levels: Sequence[str]
    state: Dict[str, Any]


def _score_request(spec: SystemOneScoreSpec):
    return _jsonable(spec.state), {
        "verdict": ScoreQuestion(
            instructions=spec.instructions, levels=list(spec.levels)
        )
    }


def _book(metric: Any, result: Optional[Any]) -> Optional[Any]:
    if result is None:
        return None
    answers, cost = result
    _accrue(metric, cost)
    _record_confidence(metric, answers)
    return answers["verdict"]


def system_one_probability(
    metric: Metric, spec: Optional[SystemOneBinarySpec]
) -> Optional[float]:
    """Jev's P(yes) for one yes/no decision, or `None` for the LLM."""
    if not _system_one_active(metric, spec):
        return None
    answer = _book(
        metric,
        _system_one_call(
            metric, metric.system_one_model.noul, *_binary_request(spec)
        ),
    )
    return None if answer is None else answer.probability


async def a_system_one_probability(
    metric: Metric, spec: Optional[SystemOneBinarySpec]
) -> Optional[float]:
    if not _system_one_active(metric, spec):
        return None
    answer = _book(
        metric,
        await _a_system_one_call(
            metric, metric.system_one_model.a_noul, *_binary_request(spec)
        ),
    )
    return None if answer is None else answer.probability


def system_one_score(
    metric: Metric, spec: Optional[SystemOneScoreSpec]
) -> Optional[float]:
    """Jev's score on `spec.levels`, mapped onto `[0, 1]`, or `None` for the
    LLM."""
    if not _system_one_active(metric, spec):
        return None
    answer = _book(
        metric,
        _system_one_call(
            metric, metric.system_one_model.score, *_score_request(spec)
        ),
    )
    return None if answer is None else answer.normalized


async def a_system_one_score(
    metric: Metric, spec: Optional[SystemOneScoreSpec]
) -> Optional[float]:
    if not _system_one_active(metric, spec):
        return None
    answer = _book(
        metric,
        await _a_system_one_call(
            metric, metric.system_one_model.a_score, *_score_request(spec)
        ),
    )
    return None if answer is None else answer.normalized


def format_decision_reason(metric: Metric, what: str, value: float) -> str:
    """The reason that stands in for the LLM's when Jev made a decision the
    LLM would have explained in the same response."""
    model = getattr(metric, "system_one_model", None)
    name = model.get_model_name() if model is not None else "System One"
    text = f"Decided by {name}: {what} {value:.2f}"
    confidence = getattr(metric, "confidence", None)
    if confidence is not None:
        text += f", confidence {confidence:.2f}"
    return text + "."


###############################################
# Rubric score (G-Eval)
###############################################
#
# G-Eval is LLM-as-a-judge by definition: the LLM drafts the evaluation
# steps and produces the score, smoothed by token probabilities. It never
# asks a System One model (use `JevEval` for a Jev-native custom metric), so
# these helpers have no `system_one` branch.


def _no_log_prob_support(model) -> bool:
    from deepeval.metrics.g_eval.utils import no_log_prob_support

    return no_log_prob_support(model)


def _score_from_raw_response(
    metric: Metric,
    res: Any,
    cost: Any,
    *,
    strict_mode: bool,
    weighted_score_fn: Callable[[int, Any], Union[int, float]],
) -> Tuple[Union[int, float], str]:
    _accrue(metric, cost)
    data = trimAndLoadJson(res.choices[0].message.content, metric)
    reason = data["reason"]
    score = data["score"]
    if strict_mode:
        return score, reason
    try:
        return weighted_score_fn(score, res), reason
    except (KeyError, AttributeError, TypeError, ValueError):
        return score, reason


def generate_rubric_score(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[Any],
    strict_mode: bool,
    top_logprobs: int,
    weighted_score_fn: Callable[[int, Any], Union[int, float]],
) -> Tuple[Union[int, float], str]:
    try:
        if _no_log_prob_support(metric.model):
            raise AttributeError("log_probs unsupported.")
        res, cost = metric.model.generate_raw_response(
            prompt, top_logprobs=top_logprobs
        )
        return _score_from_raw_response(
            metric,
            res,
            cost,
            strict_mode=strict_mode,
            weighted_score_fn=weighted_score_fn,
        )
    except AttributeError:
        return generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=schema_cls,
            extract_schema=lambda s: (s.score, s.reason),
            extract_json=lambda d: (d["score"], d["reason"]),
        )


async def a_generate_rubric_score(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[Any],
    strict_mode: bool,
    top_logprobs: int,
    weighted_score_fn: Callable[[int, Any], Union[int, float]],
) -> Tuple[Union[int, float], str]:
    try:
        if _no_log_prob_support(metric.model):
            raise AttributeError("log_probs unsupported.")
        res, cost = await metric.model.a_generate_raw_response(
            prompt, top_logprobs=top_logprobs
        )
        return _score_from_raw_response(
            metric,
            res,
            cost,
            strict_mode=strict_mode,
            weighted_score_fn=weighted_score_fn,
        )
    except AttributeError:
        return await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=prompt,
            schema_cls=schema_cls,
            extract_schema=lambda s: (s.score, s.reason),
            extract_json=lambda d: (d["score"], d["reason"]),
        )
