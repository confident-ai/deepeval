import json
import threading
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier.schema import (
    ClassificationResult,
    LabelMergeResult,
)
from deepeval.classifiers.utils import (
    LabelOrigin,
    ResolvedLabel,
    clean_reason,
)
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import Classifier

_MERGE_EXTRACTION = dict(
    schema_cls=LabelMergeResult,
    extract_schema=lambda result: result.same_as,
    extract_json=lambda data: data.get("same_as"),
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


@dataclass(frozen=True)
class MergedLabels:
    """The new label names the judge proposed, each mapped to the label it
    became after merging: an existing label, or itself once added to the
    pool. Names over ``max_new_labels`` are absent."""

    by_name: Dict[str, ResolvedLabel] = field(default_factory=dict)

    def get(self, name: Optional[str]) -> Optional[ResolvedLabel]:
        return None if name is None else self.by_name.get(name)


def merge_new_labels(
    classifier: "Classifier",
    classification_results: List[ClassificationResult],
) -> MergedLabels:
    config = classifier.label_discovery_config
    merged: Dict[str, ResolvedLabel] = {}
    added = 0
    for proposed, reason in _proposed_new_labels(
        classifier, classification_results
    ):
        may_add = (
            config is None
            or config.max_new_labels is None
            or added < config.max_new_labels
        )
        label = _merge(classifier, proposed, reason, may_add)
        if label is None:
            continue
        added += label is proposed
        merged[proposed.name] = _resolved(classifier, label)
    return MergedLabels(merged)


async def a_merge_new_labels(
    classifier: "Classifier",
    classification_results: List[ClassificationResult],
) -> MergedLabels:
    config = classifier.label_discovery_config
    merged: Dict[str, ResolvedLabel] = {}
    added = 0
    for proposed, reason in _proposed_new_labels(
        classifier, classification_results
    ):
        may_add = (
            config is None
            or config.max_new_labels is None
            or added < config.max_new_labels
        )
        label = await _a_merge(classifier, proposed, reason, may_add)
        if label is None:
            continue
        added += label is proposed
        merged[proposed.name] = _resolved(classifier, label)
    return MergedLabels(merged)


def _resolved(classifier: "Classifier", label: Label) -> ResolvedLabel:
    if any(declared.name == label.name for declared in classifier.labels):
        return ResolvedLabel(label.name, LabelOrigin.DECLARED)
    return ResolvedLabel(label.name, LabelOrigin.CANDIDATE, label.description)


def _merge(
    classifier: "Classifier", proposed: Label, reason: str, may_add: bool
) -> Optional[Label]:
    pool = classifier.label_pool
    labels = pool.labels
    declared = {label.name for label in classifier.labels}
    candidates = [
        *classifier.labels,
        *(label for label in labels if label.name not in declared),
    ]
    seen = len(labels)
    while True:
        match = _same_name(proposed.name, candidates)
        if match is None and candidates:
            missing = [
                label.name
                for label in [proposed, *candidates]
                if label.name not in pool.vectors
            ]
            if missing:
                pool.vectors.update(
                    zip(
                        missing,
                        classifier.label_discovery_config.embedding_model.embed_texts(
                            missing
                        ),
                    )
                )
            close_match, shortlist = _shortlist(
                classifier, proposed, candidates
            )
            if close_match is not None:
                return close_match
            same_as = generate_with_schema_and_extract(
                metric=classifier,
                prompt=_merge_prompt(classifier, proposed, reason, shortlist),
                **_MERGE_EXTRACTION,
            )
            match = next(
                (label for label in shortlist if label.name == same_as), None
            )
        if match is not None:
            return match
        if not may_add:
            return None
        additions = pool.add(proposed, seen)
        if not additions:
            return proposed
        candidates, seen = additions, seen + len(additions)


async def _a_merge(
    classifier: "Classifier", proposed: Label, reason: str, may_add: bool
) -> Optional[Label]:
    pool = classifier.label_pool
    labels = pool.labels
    declared = {label.name for label in classifier.labels}
    candidates = [
        *classifier.labels,
        *(label for label in labels if label.name not in declared),
    ]
    seen = len(labels)
    while True:
        match = _same_name(proposed.name, candidates)
        if match is None and candidates:
            missing = [
                label.name
                for label in [proposed, *candidates]
                if label.name not in pool.vectors
            ]
            if missing:
                pool.vectors.update(
                    zip(
                        missing,
                        await classifier.label_discovery_config.embedding_model.a_embed_texts(
                            missing
                        ),
                    )
                )
            close_match, shortlist = _shortlist(
                classifier, proposed, candidates
            )
            if close_match is not None:
                return close_match
            same_as = await a_generate_with_schema_and_extract(
                metric=classifier,
                prompt=_merge_prompt(classifier, proposed, reason, shortlist),
                **_MERGE_EXTRACTION,
            )
            match = next(
                (label for label in shortlist if label.name == same_as), None
            )
        if match is not None:
            return match
        if not may_add:
            return None
        additions = pool.add(proposed, seen)
        if not additions:
            return proposed
        candidates, seen = additions, seen + len(additions)


def _shortlist(
    classifier: "Classifier", proposed: Label, candidates: List[Label]
) -> Tuple[Optional[Label], List[Label]]:
    config = classifier.label_discovery_config
    vectors = classifier.label_pool.vectors
    matrix = np.array(
        [vectors[label.name] for label in [proposed, *candidates]], dtype=float
    )
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    unit = matrix / norms[:, None]
    scores = unit[1:] @ unit[0]
    order = np.argsort(-scores)
    if scores[order[0]] >= config.similarity_threshold:
        return candidates[order[0]], []
    return None, [candidates[i] for i in order[: config.top_k]]


def _proposed_new_labels(
    classifier: "Classifier",
    classification_results: List[ClassificationResult],
) -> List[Tuple[Label, str]]:
    if not classifier.label_discovery_config:
        return []
    proposed: Dict[str, Tuple[Label, str]] = {}
    for classification_result in classification_results:
        resolved_label = classifier._resolve_label_from_classification_result(
            classification_result, clean_reason(classification_result.reason)
        )
        if (
            resolved_label.origin is LabelOrigin.NEW
            and resolved_label.name is not None
            and resolved_label.name not in proposed
        ):
            proposed[resolved_label.name] = (
                Label(
                    name=resolved_label.name,
                    description=resolved_label.description,
                ),
                classification_result.reason,
            )
    return list(proposed.values())


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
                "classifier": {
                    "name": classifier.name,
                    "description": classifier.description,
                },
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


def _same_name(name: str, labels: Sequence[Label]) -> Optional[Label]:
    lowered = name.strip().lower()
    return next(
        (label for label in labels if label.name.lower() == lowered), None
    )
