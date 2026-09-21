import re
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

from deepeval.errors import DeepEvalError
from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
    Verdict,
    YES_NO,
    LEGACY_VERDICT_ALIASES,
)
from deepeval.models.system_one.schema import NoulQuestion

from .decision import (
    SYSTEM_ONE_YES_THRESHOLD,
    _jsonable,
    _system_one_active,
)
from .generation import (
    SchemaType,
    accrue_token_usage,
    generate_with_schema_and_extract,
    a_generate_with_schema_and_extract,
)

###############################################
# QAG (question-answer generation) verdicts
###############################################
#
# Every yes/no-style metric follows the same loop: ask the judge to classify a
# list of items, get back ``{"verdicts": [{"verdict": ..., "reason": ...}]}``,
# and score the fraction that land in a "passing" bucket. The functions below
# are the single implementation of that loop. Metrics only choose:
#
# - the vocabulary: ``YES_NO`` or ``YES_NO_BORDERLINE`` (``base_metric``)
# - the polarity: which verdicts count as passing when scoring
#
# ``generate_qag_verdicts`` is *the* QAG entry point.


def normalize_qag_verdict(
    raw: Any, allowed: Tuple[Verdict, ...] = YES_NO
) -> Optional[Verdict]:
    """Coerce a judge's verdict into a ``Verdict`` within ``allowed``.

    Judges don't always honor a strict instruction — a non-native model may
    reply "No, it's off-topic". This strips, lowercases and keeps only the
    leading word, maps legacy aliases (``idk`` -> ``Verdict.BORDERLINE``), and
    returns ``None`` for anything still outside ``allowed`` so the caller can
    drop it from scoring rather than miscount it.
    """
    if isinstance(raw, Verdict):
        return raw if raw in allowed else None
    match = re.match(
        r"[a-z]+", str(raw if raw is not None else "").strip().lower()
    )
    leading = match.group(0) if match else ""
    verdict = LEGACY_VERDICT_ALIASES.get(leading)
    if verdict is None:
        try:
            verdict = Verdict(leading)
        except ValueError:
            return None
    return verdict if verdict in allowed else None


def verdict_from_json(
    data: Dict,
    verdict_cls: Type[SchemaType],
    allowed: Tuple[Verdict, ...] = YES_NO,
) -> Optional[SchemaType]:
    """Build a single verdict object from a judge's loose JSON output.

    Returns ``None`` when the verdict is outside ``allowed``. Extra keys on
    ``data`` (e.g. ``statement``) are forwarded to ``verdict_cls``.
    """
    normalized = normalize_qag_verdict(data.get("verdict"), allowed)
    if normalized is None:
        return None
    return verdict_cls(**{**data, "verdict": normalized})


def _verdicts_from_json(
    data: Dict,
    verdict_cls: Type[SchemaType],
    allowed: Tuple[Verdict, ...],
) -> List[SchemaType]:
    verdicts: List[SchemaType] = []
    for item in data.get("verdicts", []) or []:
        verdict = verdict_from_json(item, verdict_cls, allowed)
        if verdict is not None:
            verdicts.append(verdict)
    return verdicts


def score_qag_verdicts(
    metric: Union[BaseMetric, BaseConversationalMetric],
    verdicts: List[Any],
    *,
    passing: Tuple[Verdict, ...],
    empty_score: float = 1,
) -> float:
    """Fraction of ``verdicts`` whose ``.verdict`` is in ``passing``.

    ``passing`` sets the polarity: ``(Verdict.YES,)`` for "yes is good" metrics
    such as contextual recall, ``(Verdict.NO,)`` for "yes is bad" metrics such
    as bias, ``(Verdict.YES, Verdict.BORDERLINE)`` to let borderline items pass. ``None`` entries
    (dropped out-of-vocabulary verdicts) are ignored. Applies the shared
    ``strict_mode`` clamp.
    """
    verdicts = [v for v in verdicts if v is not None]
    if len(verdicts) == 0:
        return empty_score

    passed = sum(
        1
        for v in verdicts
        if normalize_qag_verdict(v.verdict, passing) is not None
    )
    score = passed / len(verdicts)
    return 0 if metric.strict_mode and score < metric.threshold else score


###############################################
# System One (Jev) verdicts
###############################################
#
# Under DEEPEVAL_MODE=experimental the decision step of a QAG metric is a set
# of Noul questions, one per item, answered by a System One model. The LLM
# still extracts the items and writes the reasons. P(yes) is thresholded into
# the metric's verdict vocabulary; there is deliberately no LLM fallback.

SYSTEM_ONE_BORDERLINE_LOW = 0.35
SYSTEM_ONE_BORDERLINE_HIGH = 0.65


@dataclass
class SystemOneVerdictSpec:
    instructions: str
    items: Sequence[Any]
    item_key: str
    state: Optional[Dict[str, Any]] = None
    criteria: Optional[Tuple[Any, Any]] = None
    build_verdict: Optional[Callable[[Any, Verdict, float], Any]] = None


def verdict_from_probability(
    probability: float, allowed: Tuple[Verdict, ...] = YES_NO
) -> Verdict:
    if Verdict.BORDERLINE in allowed:
        if probability > SYSTEM_ONE_BORDERLINE_HIGH:
            return Verdict.YES
        if probability < SYSTEM_ONE_BORDERLINE_LOW:
            return Verdict.NO
        return Verdict.BORDERLINE
    return (
        Verdict.YES if probability >= SYSTEM_ONE_YES_THRESHOLD else Verdict.NO
    )


def _system_one_request(
    spec: SystemOneVerdictSpec,
) -> Tuple[Dict[str, Any], Dict[str, NoulQuestion]]:
    items = [_jsonable(item) for item in spec.items]
    state = {**_jsonable(spec.state or {}), f"{spec.item_key}s": items}
    true, false = spec.criteria if spec.criteria else (None, None)
    questions = {
        f"{spec.item_key}_{i}": NoulQuestion(
            instructions={spec.item_key: item, "question": spec.instructions},
            true=true,
            false=false,
        )
        for i, item in enumerate(items)
    }
    return state, questions


def _system_one_verdicts(
    spec: SystemOneVerdictSpec,
    answers: Dict[str, Any],
    verdict_cls: Type[SchemaType],
    allowed: Tuple[Verdict, ...],
) -> List[SchemaType]:
    verdicts: List[SchemaType] = []
    for i, item in enumerate(spec.items):
        answer = answers.get(f"{spec.item_key}_{i}")
        if answer is None:
            raise DeepEvalError(
                f"System One model returned no answer for {spec.item_key} {i}."
            )
        p = answer.probability
        verdict = verdict_from_probability(p, allowed)
        if spec.build_verdict is not None:
            verdicts.append(spec.build_verdict(item, verdict, p))
        else:
            verdicts.append(
                verdict_cls(verdict=verdict, reason=f"P(yes)={p:.2f}")
            )
    return verdicts


def generate_qag_verdicts(
    metric: Union[BaseMetric, BaseConversationalMetric],
    prompt: Any,
    *,
    verdict_cls: Type[SchemaType],
    verdicts_cls: Type[Any],
    allowed: Tuple[Verdict, ...] = YES_NO,
    system_one: Optional[SystemOneVerdictSpec] = None,
) -> List[SchemaType]:
    """THE QAG entry point: ask the judge for a list of verdicts.

    ``verdicts_cls`` is the wrapper schema with a ``verdicts: List[verdict_cls]``
    field. On the strict-schema path the ``Literal`` on ``verdict_cls`` already
    enforces the vocabulary; on the loose-JSON path each item is normalized via
    ``normalize_qag_verdict`` and out-of-vocabulary items are dropped. Score the
    result with ``score_qag_verdicts``.

    ``system_one`` describes the same decision as Noul questions; it is used
    instead of ``prompt`` when ``DEEPEVAL_MODE=experimental``.
    """
    if _system_one_active(metric, system_one):
        if len(system_one.items) == 0:
            return []
        state, questions = _system_one_request(system_one)
        answers, cost = metric.system_one_model.noul(state, questions)
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
        return _system_one_verdicts(system_one, answers, verdict_cls, allowed)

    return generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=verdicts_cls,
        extract_schema=lambda s: list(s.verdicts),
        extract_json=lambda data: _verdicts_from_json(
            data, verdict_cls, allowed
        ),
    )


async def a_generate_qag_verdicts(
    metric: Union[BaseMetric, BaseConversationalMetric],
    prompt: Any,
    *,
    verdict_cls: Type[SchemaType],
    verdicts_cls: Type[Any],
    allowed: Tuple[Verdict, ...] = YES_NO,
    system_one: Optional[SystemOneVerdictSpec] = None,
) -> List[SchemaType]:
    """Async counterpart of ``generate_qag_verdicts``."""
    if _system_one_active(metric, system_one):
        if len(system_one.items) == 0:
            return []
        state, questions = _system_one_request(system_one)
        answers, cost = await metric.system_one_model.a_noul(state, questions)
        metric._accrue_cost(cost)
        accrue_token_usage(metric, cost)
        return _system_one_verdicts(system_one, answers, verdict_cls, allowed)

    return await a_generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=verdicts_cls,
        extract_schema=lambda s: list(s.verdicts),
        extract_json=lambda data: _verdicts_from_json(
            data, verdict_cls, allowed
        ),
    )


def generate_qag_verdict(
    metric: Union[BaseMetric, BaseConversationalMetric],
    prompt: Any,
    *,
    verdict_cls: Type[SchemaType],
    allowed: Tuple[Verdict, ...] = YES_NO,
) -> Optional[SchemaType]:
    """Single-verdict variant of ``generate_qag_verdicts``.

    Returns ``None`` when the loose-JSON verdict is out of vocabulary.
    """
    return generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=verdict_cls,
        extract_schema=lambda s: s,
        extract_json=lambda data: verdict_from_json(data, verdict_cls, allowed),
    )


async def a_generate_qag_verdict(
    metric: Union[BaseMetric, BaseConversationalMetric],
    prompt: Any,
    *,
    verdict_cls: Type[SchemaType],
    allowed: Tuple[Verdict, ...] = YES_NO,
) -> Optional[SchemaType]:
    """Async counterpart of ``generate_qag_verdict``."""
    return await a_generate_with_schema_and_extract(
        metric=metric,
        prompt=prompt,
        schema_cls=verdict_cls,
        extract_schema=lambda s: s,
        extract_json=lambda data: verdict_from_json(data, verdict_cls, allowed),
    )
