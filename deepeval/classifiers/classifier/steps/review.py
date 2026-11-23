import asyncio
import json
import logging
from typing import TYPE_CHECKING, List, Optional

from deepeval.classifiers.classifier.schema import LabelReview
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_run.api import Classification

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import Classifier

logger = logging.getLogger(__name__)


def review_classifications(classifier: "Classifier") -> None:
    _apply_reviews(
        classifier,
        [_review(classifier, c) for c in classifier.classifications],
    )


async def a_review_classifications(classifier: "Classifier") -> None:
    _apply_reviews(
        classifier,
        await asyncio.gather(
            *(_a_review(classifier, c) for c in classifier.classifications)
        ),
    )


def _apply_reviews(
    classifier: "Classifier", reviews: List[Optional[LabelReview]]
) -> None:
    kept, dropped = [], []
    for classification, review in zip(classifier.classifications, reviews):
        if review is not None:
            classification.confidence = review.confidence
        if review is None or classifier._passes_review(review):
            kept.append(classification)
        else:
            dropped.append((classification, review))
    kept.sort(key=lambda c: -1.0 if c.confidence is None else -c.confidence)
    classifier.classifications = kept
    top = kept[0] if kept else None
    classifier.label = top.label if top else None
    if top is not None:
        classifier.reason = top.reason
    elif dropped and classifier.include_reason:
        classification, review = max(
            dropped, key=lambda pair: pair[1].confidence
        )
        classifier.reason = " ".join(
            part
            for part in (
                f'Proposed "{classification.label}" but review dropped it.',
                classifier._review_reason(review),
            )
            if part
        )


def _review(
    classifier: "Classifier", classification: Classification
) -> Optional[LabelReview]:
    schema = classifier._review_schema()
    try:
        return generate_with_schema_and_extract(
            metric=classifier,
            prompt=_review_prompt(classifier, classification),
            schema_cls=schema,
            extract_schema=lambda review: review,
            extract_json=schema.model_validate,
        )
    except ValueError as error:
        logger.warning(
            f"Unreadable review; keeping {classification.label!r}: {error}"
        )
        return None


async def _a_review(
    classifier: "Classifier", classification: Classification
) -> Optional[LabelReview]:
    schema = classifier._review_schema()
    try:
        return await a_generate_with_schema_and_extract(
            metric=classifier,
            prompt=_review_prompt(classifier, classification),
            schema_cls=schema,
            extract_schema=lambda review: review,
            extract_json=schema.model_validate,
        )
    except ValueError as error:
        logger.warning(
            f"Unreadable review; keeping {classification.label!r}: {error}"
        )
        return None


def _review_prompt(
    classifier: "Classifier", classification: Classification
) -> str:
    subject, content = classifier._review_content
    context = classifier.classification_context
    app_description = context.app_description if context else None
    description = next(
        (
            label.description
            for label in classifier.labels
            if label.name == classification.label
        ),
        classification.new_label_description,
    )
    return classifier._get_prompt(
        "review_label",
        subject=subject,
        label=json.dumps(
            {
                "name": classification.label,
                "description": description,
                "passages": [item.quote for item in classification.evidence],
            },
            ensure_ascii=False,
        ),
        content=content,
        app_description=app_description,
        app_note=(
            classifier._get_prompt("review_app_note") if app_description else ""
        ),
    )
