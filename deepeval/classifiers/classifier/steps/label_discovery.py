import json
import logging
import re
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from deepeval.classifiers.base_classifier import NONE_LABEL, Label
from deepeval.classifiers.classifier.schema import LabelMergeResult
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import Classifier

logger = logging.getLogger(__name__)

_MERGE_EXTRACTION = dict(
    schema_cls=LabelMergeResult,
    extract_schema=lambda result: result.same_as,
    extract_json=lambda data: data.get("same_as"),
)


def merge_new_labels(classifier: "Classifier", result: Any) -> Dict[str, Label]:
    return {
        name: _merge(
            classifier, Label(name=name, description=description), reason
        )
        for name, description, reason in _proposed_new_labels(
            classifier, result
        )
    }


async def a_merge_new_labels(
    classifier: "Classifier", result: Any
) -> Dict[str, Label]:
    return {
        name: await _a_merge(
            classifier, Label(name=name, description=description), reason
        )
        for name, description, reason in _proposed_new_labels(
            classifier, result
        )
    }


def merged_label(
    classifier: "Classifier", match: Label
) -> Tuple[str, Optional[str]]:
    if any(label.name == match.name for label in classifier.labels):
        return match.name, None
    return match.name, match.description


def is_new_label(classifier: "Classifier", label: str) -> bool:
    return (
        classifier.label_discovery_config is not None
        and same_name(label, classifier.label_pool.labels) is None
    )


class LabelPool:
    def __init__(self, labels: Sequence[Label] = ()):
        self._labels: List[Label] = list(labels)
        self.vectors: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    @property
    def labels(self) -> List[Label]:
        with self._lock:
            return list(self._labels)

    def add(self, label: Label, seen: int) -> List[Label]:
        with self._lock:
            additions = self._labels[seen:]
            if not additions:
                self._labels.append(label)
            return list(additions)


def _merge(classifier: "Classifier", proposed: Label, reason: str) -> Label:
    candidates, seen = _merge_targets(classifier)
    while True:
        match = _find_match(classifier, proposed, reason, candidates)
        if match is not None:
            return match
        additions = classifier.label_pool.add(proposed, seen)
        if not additions:
            return proposed
        candidates, seen = additions, seen + len(additions)


async def _a_merge(
    classifier: "Classifier", proposed: Label, reason: str
) -> Label:
    candidates, seen = _merge_targets(classifier)
    while True:
        match = await _a_find_match(classifier, proposed, reason, candidates)
        if match is not None:
            return match
        additions = classifier.label_pool.add(proposed, seen)
        if not additions:
            return proposed
        candidates, seen = additions, seen + len(additions)


def _find_match(
    classifier: "Classifier",
    proposed: Label,
    reason: str,
    candidates: List[Label],
) -> Optional[Label]:
    exact = same_name(proposed.name, candidates)
    if exact is not None or not candidates:
        return exact
    shortlist = _shortlist(
        classifier,
        proposed,
        candidates,
        _embed(classifier, [proposed, *candidates]),
    )
    if isinstance(shortlist, Label):
        return shortlist
    same_as = generate_with_schema_and_extract(
        metric=classifier,
        prompt=_merge_prompt(classifier, proposed, reason, shortlist),
        **_MERGE_EXTRACTION,
    )
    return next((label for label in shortlist if label.name == same_as), None)


async def _a_find_match(
    classifier: "Classifier",
    proposed: Label,
    reason: str,
    candidates: List[Label],
) -> Optional[Label]:
    exact = same_name(proposed.name, candidates)
    if exact is not None or not candidates:
        return exact
    shortlist = _shortlist(
        classifier,
        proposed,
        candidates,
        await _a_embed(classifier, [proposed, *candidates]),
    )
    if isinstance(shortlist, Label):
        return shortlist
    same_as = await a_generate_with_schema_and_extract(
        metric=classifier,
        prompt=_merge_prompt(classifier, proposed, reason, shortlist),
        **_MERGE_EXTRACTION,
    )
    return next((label for label in shortlist if label.name == same_as), None)


def _shortlist(
    classifier: "Classifier",
    proposed: Label,
    candidates: List[Label],
    vectors: Optional[List[List[float]]],
) -> Union[Label, List[Label]]:
    config = classifier.label_discovery_config
    if vectors is None:
        words = _words(proposed.name)
        return sorted(
            candidates,
            key=lambda label: len(words & _words(label.name)),
            reverse=True,
        )[: config.top_k]
    ranked = sorted(
        zip(
            (_cosine(vectors[0], vector) for vector in vectors[1:]), candidates
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if ranked[0][0] >= config.similarity_threshold:
        return ranked[0][1]
    return [label for _, label in ranked[: config.top_k]]


def _proposed_new_labels(
    classifier: "Classifier", result: Any
) -> List[Tuple[str, str, str]]:
    if not classifier.label_discovery_config:
        return []
    new = []
    for item in result.labels if classifier.multi_label else [result]:
        label, description = classifier._resolve(
            item.label,
            item.reason,
            getattr(item, "new_label_description", None),
        )
        if (
            label not in (None, NONE_LABEL)
            and description is not None
            and same_name(label, classifier.label_pool.labels) is None
            and all(label != name for name, _, _ in new)
        ):
            new.append((label, description, item.reason))
    cap = classifier.label_discovery_config.max_new_labels
    return new if cap is None else new[:cap]


def _merge_prompt(
    classifier: "Classifier",
    proposed: Label,
    reason: str,
    candidates: List[Label],
) -> str:
    return classifier._get_prompt(
        "merge_labels",
        payload=json.dumps(
            {
                "classifier": {"name": classifier.name},
                "proposed": {
                    "name": proposed.name,
                    "description": proposed.description,
                    "reason": reason,
                },
                "candidates": [
                    {"name": label.name, "description": label.description}
                    for label in candidates
                ],
            },
            ensure_ascii=False,
        ),
    )


def _embed(
    classifier: "Classifier", labels: List[Label]
) -> Optional[List[List[float]]]:
    model = classifier.label_discovery_config.embedding_model
    if model is None:
        return None
    vectors = classifier.label_pool.vectors
    missing = [label.name for label in labels if label.name not in vectors]
    if missing:
        try:
            vectors.update(zip(missing, model.embed_texts(missing)))
        except Exception as error:
            logger.warning(f"Label embedding failed; comparing names: {error}")
            return None
    return [vectors[label.name] for label in labels]


async def _a_embed(
    classifier: "Classifier", labels: List[Label]
) -> Optional[List[List[float]]]:
    model = classifier.label_discovery_config.embedding_model
    if model is None:
        return None
    vectors = classifier.label_pool.vectors
    missing = [label.name for label in labels if label.name not in vectors]
    if missing:
        try:
            vectors.update(zip(missing, await model.a_embed_texts(missing)))
        except Exception as error:
            logger.warning(f"Label embedding failed; comparing names: {error}")
            return None
    return [vectors[label.name] for label in labels]


def _merge_targets(classifier: "Classifier") -> Tuple[List[Label], int]:
    pool = classifier.label_pool.labels
    names = {label.name for label in classifier.labels}
    targets = [*classifier.labels, *(l for l in pool if l.name not in names)]
    return targets, len(pool)


def same_name(name: str, labels: Sequence[Label]) -> Optional[Label]:
    lowered = name.strip().lower()
    return next(
        (label for label in labels if label.name.lower() == lowered), None
    )


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def _words(name: str) -> Set[str]:
    return set(re.findall(r"\w+", name.lower()))
