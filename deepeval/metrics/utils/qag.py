import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from deepeval.metrics.base_metric import (
    BaseMetric,
    BaseConversationalMetric,
    Verdict,
    YES_NO,
    LEGACY_VERDICT_ALIASES,
)

from .generation import (
    SchemaType,
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


def generate_qag_verdicts(
    metric: Union[BaseMetric, BaseConversationalMetric],
    prompt: Any,
    *,
    verdict_cls: Type[SchemaType],
    verdicts_cls: Type[Any],
    allowed: Tuple[Verdict, ...] = YES_NO,
) -> List[SchemaType]:
    """THE QAG entry point: ask the judge for a list of verdicts.

    ``verdicts_cls`` is the wrapper schema with a ``verdicts: List[verdict_cls]``
    field. On the strict-schema path the ``Literal`` on ``verdict_cls`` already
    enforces the vocabulary; on the loose-JSON path each item is normalized via
    ``normalize_qag_verdict`` and out-of-vocabulary items are dropped. Score the
    result with ``score_qag_verdicts``.
    """
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
) -> List[SchemaType]:
    """Async counterpart of ``generate_qag_verdicts``."""
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
