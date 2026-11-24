import json
import logging
import re
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

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier.schema import (
    ClassificationResult,
    LabelMergeResult,
)
from deepeval.classifiers.classifier.steps.classification import (
    MAX_GENERATION_ATTEMPTS,
    PROVIDER_ERRORS,
)
from deepeval.classifiers.utils import (
    LabelOrigin,
    ResolvedLabel,
    clean_reason,
)
from deepeval.errors import DeepEvalError
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


class LabelPool:
    def __init__(self, labels: Sequence[Label] = ()):
        self._labels: List[Label] = list(labels)
        self.vectors: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    @property
    def labels(self) -> List[Label]:
        with self._lock:
            return list(self._labels)

    def add_if_unchanged(self, label: Label, seen_count: int) -> List[Label]:
        """Adds ``label`` only when the pool still holds the ``seen_count``
        labels the caller compared it against. Otherwise returns the labels
        added since, for the caller to compare against first."""
        with self._lock:
            labels_added_since = self._labels[seen_count:]
            if not labels_added_since:
                self._labels.append(label)
            return list(labels_added_since)


@dataclass(frozen=True)
class MergedLabels:
    """The new label names the judge proposed, each mapped to the label it
    became after merging: an existing label, or itself once added to the
    pool. Names over ``max_new_labels`` are absent."""

    by_name: Dict[str, ResolvedLabel] = field(default_factory=dict)

    def get(self, name: Optional[str]) -> Optional[ResolvedLabel]:
        return None if name is None else self.by_name.get(name)


@dataclass(frozen=True)
class LabelDiscoveryStep:
    merged_labels: Dict[str, str]
    seconds: float


def merge_new_labels(
    classifier: "Classifier",
    classification_results: List[ClassificationResult],
    subject: str,
) -> MergedLabels:
    config = classifier.label_discovery_config
    max_new_labels = config.max_new_labels if config else None
    merged_by_name: Dict[str, ResolvedLabel] = {}
    added_count = 0
    for proposed_label, reason in _collect_proposed_labels(
        classifier, classification_results, subject
    ):
        can_add_to_pool = max_new_labels is None or added_count < max_new_labels
        merged_label = _merge_or_add_label(
            classifier, proposed_label, reason, can_add_to_pool
        )
        if merged_label is None:
            continue
        added_count += merged_label is proposed_label
        merged_by_name[proposed_label.name] = _resolve_merged_label(
            classifier, merged_label
        )
    return MergedLabels(merged_by_name)


async def a_merge_new_labels(
    classifier: "Classifier",
    classification_results: List[ClassificationResult],
    subject: str,
) -> MergedLabels:
    config = classifier.label_discovery_config
    max_new_labels = config.max_new_labels if config else None
    merged_by_name: Dict[str, ResolvedLabel] = {}
    added_count = 0
    for proposed_label, reason in _collect_proposed_labels(
        classifier, classification_results, subject
    ):
        can_add_to_pool = max_new_labels is None or added_count < max_new_labels
        merged_label = await _a_merge_or_add_label(
            classifier, proposed_label, reason, can_add_to_pool
        )
        if merged_label is None:
            continue
        added_count += merged_label is proposed_label
        merged_by_name[proposed_label.name] = _resolve_merged_label(
            classifier, merged_label
        )
    return MergedLabels(merged_by_name)


def _resolve_merged_label(
    classifier: "Classifier", merged_label: Label
) -> ResolvedLabel:
    is_declared = any(
        declared_label.name == merged_label.name
        for declared_label in classifier.labels
    )
    if is_declared:
        return ResolvedLabel(merged_label.name, LabelOrigin.DECLARED)
    return ResolvedLabel(
        merged_label.name, LabelOrigin.CANDIDATE, merged_label.description
    )


def _merge_or_add_label(
    classifier: "Classifier",
    proposed_label: Label,
    reason: str,
    can_add_to_pool: bool,
) -> Optional[Label]:
    """Returns the existing label ``proposed_label`` duplicates, or
    ``proposed_label`` itself once added to the pool, or ``None`` when it
    matches nothing and may not be added."""
    pool = classifier.label_pool
    pooled_labels = pool.labels
    declared_names = {label.name for label in classifier.labels}
    existing_labels = [
        *classifier.labels,
        *(label for label in pooled_labels if label.name not in declared_names),
    ]
    seen_count = len(pooled_labels)
    while True:
        # Same name
        matching_label = _find_label_by_name(
            proposed_label.name, existing_labels
        )
        if matching_label is None and existing_labels:
            # Embedding similarity, or shared words when embedding fails
            if _embed_labels(classifier, [proposed_label, *existing_labels]):
                near_duplicate, similar_labels = _find_similar_labels(
                    classifier, proposed_label, existing_labels
                )
                if near_duplicate is not None:
                    return near_duplicate
            else:
                similar_labels = _labels_sharing_words(
                    classifier, proposed_label, existing_labels
                )

            # Merge judge
            duplicate_of = (
                _ask_merge_judge(
                    classifier,
                    _build_merge_prompt(
                        classifier, proposed_label, reason, similar_labels
                    ),
                )
                if similar_labels
                else None
            )
            matching_label = next(
                (
                    label
                    for label in similar_labels
                    if label.name == duplicate_of
                ),
                None,
            )
        if matching_label is not None:
            return matching_label

        # New label
        if not can_add_to_pool:
            return None
        labels_added_since = pool.add_if_unchanged(proposed_label, seen_count)
        if not labels_added_since:
            return proposed_label
        existing_labels = labels_added_since
        seen_count += len(labels_added_since)


async def _a_merge_or_add_label(
    classifier: "Classifier",
    proposed_label: Label,
    reason: str,
    can_add_to_pool: bool,
) -> Optional[Label]:
    """Returns the existing label ``proposed_label`` duplicates, or
    ``proposed_label`` itself once added to the pool, or ``None`` when it
    matches nothing and may not be added."""
    pool = classifier.label_pool
    pooled_labels = pool.labels
    declared_names = {label.name for label in classifier.labels}
    existing_labels = [
        *classifier.labels,
        *(label for label in pooled_labels if label.name not in declared_names),
    ]
    seen_count = len(pooled_labels)
    while True:
        # Same name
        matching_label = _find_label_by_name(
            proposed_label.name, existing_labels
        )
        if matching_label is None and existing_labels:
            # Embedding similarity, or shared words when embedding fails
            if await _a_embed_labels(
                classifier, [proposed_label, *existing_labels]
            ):
                near_duplicate, similar_labels = _find_similar_labels(
                    classifier, proposed_label, existing_labels
                )
                if near_duplicate is not None:
                    return near_duplicate
            else:
                similar_labels = _labels_sharing_words(
                    classifier, proposed_label, existing_labels
                )

            # Merge judge
            duplicate_of = (
                await _a_ask_merge_judge(
                    classifier,
                    _build_merge_prompt(
                        classifier, proposed_label, reason, similar_labels
                    ),
                )
                if similar_labels
                else None
            )
            matching_label = next(
                (
                    label
                    for label in similar_labels
                    if label.name == duplicate_of
                ),
                None,
            )
        if matching_label is not None:
            return matching_label

        # New label
        if not can_add_to_pool:
            return None
        labels_added_since = pool.add_if_unchanged(proposed_label, seen_count)
        if not labels_added_since:
            return proposed_label
        existing_labels = labels_added_since
        seen_count += len(labels_added_since)


def _ask_merge_judge(classifier: "Classifier", prompt: str) -> Optional[str]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return generate_with_schema_and_extract(
                metric=classifier, prompt=prompt, **_MERGE_EXTRACTION
            )
        except (ValueError, DeepEvalError):
            classifier.error = None
        except PROVIDER_ERRORS as error:
            logger.warning(
                f"Merge judge call failed; keeping the label: {error}"
            )
            return None
    return None


async def _a_ask_merge_judge(
    classifier: "Classifier", prompt: str
) -> Optional[str]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return await a_generate_with_schema_and_extract(
                metric=classifier, prompt=prompt, **_MERGE_EXTRACTION
            )
        except (ValueError, DeepEvalError):
            classifier.error = None
        except PROVIDER_ERRORS as error:
            logger.warning(
                f"Merge judge call failed; keeping the label: {error}"
            )
            return None
    return None


def _embed_labels(classifier: "Classifier", labels: List[Label]) -> bool:
    """Embeds the label names the pool has no vector for yet, trying twice.
    Returns whether every name now has a vector."""
    pool = classifier.label_pool
    names = [label.name for label in labels if label.name not in pool.vectors]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        if not names:
            return True
        try:
            embeddings = (
                classifier.label_discovery_config.embedding_model.embed_texts(
                    names
                )
            )
        except PROVIDER_ERRORS as error:
            logger.warning(f"Label embedding failed: {error}")
            continue
        if len(embeddings) == len(names):
            pool.vectors.update(zip(names, embeddings))
            return True
        logger.warning(
            f"The embedding model returned {len(embeddings)} vectors for "
            f"{len(names)} label names."
        )
    return False


async def _a_embed_labels(
    classifier: "Classifier", labels: List[Label]
) -> bool:
    """Embeds the label names the pool has no vector for yet, trying twice.
    Returns whether every name now has a vector."""
    pool = classifier.label_pool
    names = [label.name for label in labels if label.name not in pool.vectors]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        if not names:
            return True
        try:
            embeddings = await classifier.label_discovery_config.embedding_model.a_embed_texts(
                names
            )
        except PROVIDER_ERRORS as error:
            logger.warning(f"Label embedding failed: {error}")
            continue
        if len(embeddings) == len(names):
            pool.vectors.update(zip(names, embeddings))
            return True
        logger.warning(
            f"The embedding model returned {len(embeddings)} vectors for "
            f"{len(names)} label names."
        )
    return False


def _labels_sharing_words(
    classifier: "Classifier",
    proposed_label: Label,
    existing_labels: List[Label],
) -> List[Label]:
    """The ``top_k`` existing labels sharing the most words with the proposed
    name, for the merge judge to compare when there are no embeddings."""
    words = set(re.findall(r"\w+", proposed_label.name.lower()))
    ranked = sorted(
        existing_labels,
        key=lambda label: len(
            words & set(re.findall(r"\w+", label.name.lower()))
        ),
        reverse=True,
    )
    return ranked[: classifier.label_discovery_config.top_k]


def _find_similar_labels(
    classifier: "Classifier",
    proposed_label: Label,
    existing_labels: List[Label],
) -> Tuple[Optional[Label], List[Label]]:
    """Returns ``(near_duplicate, similar_labels)``: an existing label at or
    above ``similarity_threshold``, or else the ``top_k`` closest labels."""
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required for label discovery. Install it with "
            f"`pip install numpy`. Root cause: {e}"
        )

    config = classifier.label_discovery_config
    vectors = classifier.label_pool.vectors
    embeddings = np.array(
        [vectors[label.name] for label in [proposed_label, *existing_labels]],
        dtype=float,
    )
    norms = np.linalg.norm(embeddings, axis=1)
    norms[norms == 0] = 1
    unit_embeddings = embeddings / norms[:, None]
    similarities = unit_embeddings[1:] @ unit_embeddings[0]
    most_similar_first = np.argsort(-similarities)
    closest = most_similar_first[0]
    if similarities[closest] >= config.similarity_threshold:
        return existing_labels[closest], []
    return None, [
        existing_labels[index] for index in most_similar_first[: config.top_k]
    ]


def _collect_proposed_labels(
    classifier: "Classifier",
    classification_results: List[ClassificationResult],
    subject: str,
) -> List[Tuple[Label, str]]:
    if not classifier.label_discovery_config:
        return []
    proposed_by_name: Dict[str, Tuple[Label, str]] = {}
    for classification_result in classification_results:
        resolved_label = classifier._resolve_label_from_classification_result(
            classification_result,
            clean_reason(classification_result.reason),
            subject,
        )
        if (
            resolved_label.origin is LabelOrigin.NEW
            and resolved_label.name is not None
            and resolved_label.name not in proposed_by_name
        ):
            proposed_by_name[resolved_label.name] = (
                Label(
                    name=resolved_label.name,
                    description=resolved_label.description,
                ),
                classification_result.reason,
            )
    return list(proposed_by_name.values())


def _build_merge_prompt(
    classifier: "Classifier",
    proposed_label: Label,
    reason: str,
    similar_labels: List[Label],
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
                    "name": proposed_label.name,
                    "description": proposed_label.description,
                    "reason": reason,
                },
                "candidates": [
                    {"name": label.name, "description": label.description}
                    for label in similar_labels
                ],
            },
            ensure_ascii=False,
        ),
    )


def _find_label_by_name(name: str, labels: Sequence[Label]) -> Optional[Label]:
    lowered_name = name.strip().lower()
    return next(
        (label for label in labels if label.name.lower() == lowered_name), None
    )
