import json
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier.schema import LabelMergeResult
from deepeval.classifiers.utils import LabelOrigin
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


def _merge(classifier: "Classifier", proposed: Label, reason: str) -> Label:
    pool = classifier.label_pool
    labels = pool.labels
    declared = {label.name for label in classifier.labels}
    candidates = [
        *classifier.labels,
        *(label for label in labels if label.name not in declared),
    ]
    seen = len(labels)
    while True:
        match = same_name(proposed.name, candidates)
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
            shortlist = _shortlist(classifier, proposed, candidates)
            if isinstance(shortlist, Label):
                return shortlist
            same_as = generate_with_schema_and_extract(
                metric=classifier,
                prompt=_merge_prompt(classifier, proposed, reason, shortlist),
                **_MERGE_EXTRACTION,
            )
            match = next((l for l in shortlist if l.name == same_as), None)
        if match is not None:
            return match
        additions = pool.add(proposed, seen)
        if not additions:
            return proposed
        candidates, seen = additions, seen + len(additions)


async def _a_merge(
    classifier: "Classifier", proposed: Label, reason: str
) -> Label:
    pool = classifier.label_pool
    labels = pool.labels
    declared = {label.name for label in classifier.labels}
    candidates = [
        *classifier.labels,
        *(label for label in labels if label.name not in declared),
    ]
    seen = len(labels)
    while True:
        match = same_name(proposed.name, candidates)
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
            shortlist = _shortlist(classifier, proposed, candidates)
            if isinstance(shortlist, Label):
                return shortlist
            same_as = await a_generate_with_schema_and_extract(
                metric=classifier,
                prompt=_merge_prompt(classifier, proposed, reason, shortlist),
                **_MERGE_EXTRACTION,
            )
            match = next((l for l in shortlist if l.name == same_as), None)
        if match is not None:
            return match
        additions = pool.add(proposed, seen)
        if not additions:
            return proposed
        candidates, seen = additions, seen + len(additions)


def _shortlist(
    classifier: "Classifier", proposed: Label, candidates: List[Label]
) -> Union[Label, List[Label]]:
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
        return candidates[order[0]]
    return [candidates[i] for i in order[: config.top_k]]


def _proposed_new_labels(
    classifier: "Classifier", result: Any
) -> List[Tuple[str, str, str]]:
    if not classifier.label_discovery_config:
        return []
    new = []
    for item in result.labels if classifier.multi_label else [result]:
        resolved = classifier._resolve_item(item)
        if (
            resolved.origin is LabelOrigin.NEW
            and same_name(resolved.name, classifier.label_pool.labels) is None
            and all(resolved.name != name for name, _, _ in new)
        ):
            new.append((resolved.name, resolved.description, item.reason))
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


def same_name(name: str, labels: Sequence[Label]) -> Optional[Label]:
    lowered = name.strip().lower()
    return next(
        (label for label in labels if label.name.lower() == lowered), None
    )
