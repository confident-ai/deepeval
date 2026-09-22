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

from deepeval.config.mode import MODE_ENV_VAR, DeepEvalMode, is_experimental
from deepeval.errors import DeepEvalError
from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
)
from deepeval.models.system_one.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
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
# judge is asked. Under DEEPEVAL_MODE=experimental each accepts a
# `system_one` spec and asks a System One model instead of the LLM.

Metric = Union[BaseMetric, BaseConversationalMetric]

SYSTEM_ONE_YES_THRESHOLD = 0.5
SYSTEM_ONE_MAX_SCORE_LEVELS = 10


###############################################
# System One (Jev) plumbing, shared with qag.py
###############################################


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _system_one_active(metric: Metric, spec: Optional[Any]) -> bool:
    if spec is None or not is_experimental():
        return False
    if getattr(metric, "system_one_model", None) is None:
        raise DeepEvalError(
            f"{MODE_ENV_VAR}={DeepEvalMode.EXPERIMENTAL} routes decisions "
            f"to a System One model, but {type(metric).__name__} has none "
            f"configured. Set TYPESAFE_API_KEY or switch back with "
            f"{MODE_ENV_VAR}={DeepEvalMode.STABLE}."
        )
    return True


def _accrue(metric: Metric, cost: Any) -> None:
    metric._accrue_cost(cost)
    accrue_token_usage(metric, cost)


@dataclass
class SystemOneBinarySpec:
    instructions: Any
    state: Dict[str, Any]


@dataclass
class SystemOneChoiceSpec:
    instructions: Any
    options: Sequence[str]
    state: Dict[str, Any]


@dataclass
class SystemOneScoreSpec:
    steps: Sequence[str]
    rubric_levels: Optional[Sequence[str]]
    score_range: Tuple[int, int]
    strict_mode: bool
    state: Dict[str, Any]
    strict_instructions: str
    step_instructions: str
    rubric_instructions: str
    reason_prompt: Callable[[Union[int, float], Dict[str, float]], Any]
    reason_schema_cls: Type[Any]


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


def generate_binary_judgement(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[SchemaType],
    system_one: Optional[SystemOneBinarySpec] = None,
) -> SchemaType:
    if _system_one_active(metric, system_one):
        answers, cost = metric.system_one_model.noul(
            _jsonable(system_one.state),
            {"verdict": NoulQuestion(instructions=system_one.instructions)},
        )
        _accrue(metric, cost)
        return _binary_from_answer(schema_cls, answers["verdict"].probability)

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
        answers, cost = await metric.system_one_model.a_noul(
            _jsonable(system_one.state),
            {"verdict": NoulQuestion(instructions=system_one.instructions)},
        )
        _accrue(metric, cost)
        return _binary_from_answer(schema_cls, answers["verdict"].probability)

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
        answers, cost = metric.system_one_model.choice(
            _jsonable(system_one.state), _choice_question(system_one)
        )
        _accrue(metric, cost)
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
        answers, cost = await metric.system_one_model.a_choice(
            _jsonable(system_one.state), _choice_question(system_one)
        )
        _accrue(metric, cost)
        return _choice_from_answer(schema_cls, answers["verdict"])

    return await a_generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=schema_cls,
        extract_schema=lambda s: s,
        extract_json=lambda data: schema_cls(**data),
    )


###############################################
# Rubric score (G-Eval)
###############################################


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


def _score_state_and_questions(
    spec: SystemOneScoreSpec,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = {**_jsonable(spec.state), "evaluation_steps": list(spec.steps)}
    if spec.strict_mode:
        return state, {
            "strict": NoulQuestion(instructions=spec.strict_instructions)
        }
    if spec.rubric_levels:
        if len(spec.rubric_levels) > SYSTEM_ONE_MAX_SCORE_LEVELS:
            raise DeepEvalError(
                f"System One Score questions accept at most "
                f"{SYSTEM_ONE_MAX_SCORE_LEVELS} rubric levels; got "
                f"{len(spec.rubric_levels)}."
            )
        if len(spec.rubric_levels) < 2:
            raise DeepEvalError(
                "System One Score questions need at least 2 rubric levels."
            )
        return state, {
            "rubric": ScoreQuestion(
                instructions=spec.rubric_instructions,
                levels=list(spec.rubric_levels),
            )
        }
    return state, {
        f"step_{i}": NoulQuestion(
            instructions={"step": step, "question": spec.step_instructions}
        )
        for i, step in enumerate(spec.steps)
    }


def _score_from_answers(
    spec: SystemOneScoreSpec, answers: Any
) -> Tuple[Union[int, float], Dict[str, float]]:
    low, high = spec.score_range
    span = high - low
    if spec.strict_mode:
        p = answers.nouls["strict"].probability
        return (1 if p >= SYSTEM_ONE_YES_THRESHOLD else 0), {"strict": p}
    if spec.rubric_levels:
        answer = answers.scores["rubric"]
        top = len(spec.rubric_levels) - 1
        probabilities = {
            spec.rubric_levels[level]: p
            for level, p in answer.probabilities.items()
            if 0 <= level <= top
        }
        return low + (answer.score / top) * span, probabilities
    probabilities = {
        step: answers.nouls[f"step_{i}"].probability
        for i, step in enumerate(spec.steps)
    }
    if not probabilities:
        raise DeepEvalError("G-Eval has no evaluation steps to score.")
    mean = sum(probabilities.values()) / len(probabilities)
    return low + mean * span, probabilities


def generate_rubric_score(
    metric: Metric,
    prompt: Any,
    *,
    schema_cls: Type[Any],
    strict_mode: bool,
    top_logprobs: int,
    weighted_score_fn: Callable[[int, Any], Union[int, float]],
    system_one: Optional[SystemOneScoreSpec] = None,
) -> Tuple[Union[int, float], str]:
    if _system_one_active(metric, system_one):
        state, questions = _score_state_and_questions(system_one)
        answers, cost = metric.system_one_model.decide(state, questions)
        _accrue(metric, cost)
        score, probabilities = _score_from_answers(system_one, answers)
        reason = generate_with_schema_and_extract(
            metric=metric,
            prompt=system_one.reason_prompt(score, probabilities),
            schema_cls=system_one.reason_schema_cls,
            extract_schema=lambda s: s.reason,
            extract_json=lambda d: d["reason"],
        )
        return score, reason

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
    system_one: Optional[SystemOneScoreSpec] = None,
) -> Tuple[Union[int, float], str]:
    if _system_one_active(metric, system_one):
        state, questions = _score_state_and_questions(system_one)
        answers, cost = await metric.system_one_model.a_decide(state, questions)
        _accrue(metric, cost)
        score, probabilities = _score_from_answers(system_one, answers)
        reason = await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=system_one.reason_prompt(score, probabilities),
            schema_cls=system_one.reason_schema_cls,
            extract_schema=lambda s: s.reason,
            extract_json=lambda d: d["reason"],
        )
        return score, reason

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
