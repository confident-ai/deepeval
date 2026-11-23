import asyncio
import json
import logging
from typing import TYPE_CHECKING, List, Union

from deepeval.classifiers.classifier.schema import LabelReview
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_run.api import Classification

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import (
        ClassificationPrompt,
        Classifier,
    )

logger = logging.getLogger(__name__)


def review_classifications(
    classifier: "Classifier", prompt: "ClassificationPrompt"
) -> None:
    schema = classifier.review_schema
    reviews = []
    for classification in classifier.classifications:
        try:
            reviews.append(
                generate_with_schema_and_extract(
                    metric=classifier,
                    prompt=_build_review_prompt(
                        classifier, classification, prompt
                    ),
                    schema_cls=schema,
                    extract_schema=lambda review: review,
                    extract_json=schema.model_validate,
                )
            )
        except ValueError as error:
            reviews.append(error)
    _apply_review_results(classifier, reviews)


async def a_review_classifications(
    classifier: "Classifier", prompt: "ClassificationPrompt"
) -> None:
    schema = classifier.review_schema
    reviews = await asyncio.gather(
        *(
            a_generate_with_schema_and_extract(
                metric=classifier,
                prompt=_build_review_prompt(classifier, classification, prompt),
                schema_cls=schema,
                extract_schema=lambda review: review,
                extract_json=schema.model_validate,
            )
            for classification in classifier.classifications
        ),
        return_exceptions=True,
    )
    _apply_review_results(classifier, reviews)


def _apply_review_results(
    classifier: "Classifier", reviews: List[Union[LabelReview, BaseException]]
) -> None:
    kept_classifications, dropped_classifications = [], []
    for classification, review in zip(classifier.classifications, reviews):
        if isinstance(review, BaseException):
            if not isinstance(review, ValueError):
                raise review
            logger.warning(
                f"Unreadable review; keeping {classification.label!r}: {review}"
            )
            kept_classifications.append(classification)
            continue
        classification.confidence = review.confidence
        if review.confidence >= classifier.review_config.min_confidence:
            kept_classifications.append(classification)
        else:
            dropped_classifications.append((classification, review))
    kept_classifications.sort(
        key=lambda c: -1.0 if c.confidence is None else -c.confidence
    )
    classifier.classifications = kept_classifications
    top_classification = (
        kept_classifications[0] if kept_classifications else None
    )
    classifier.label = top_classification.label if top_classification else None
    if top_classification is not None:
        classifier.reason = top_classification.reason
    elif dropped_classifications and classifier.include_reason:
        classification, review = max(
            dropped_classifications, key=lambda pair: pair[1].confidence
        )
        classifier.reason = " ".join(
            part
            for part in (
                f'Proposed "{classification.label}" but review dropped it.',
                review.explanation(),
            )
            if part
        )


def _build_review_prompt(
    classifier: "Classifier",
    classification: Classification,
    prompt: "ClassificationPrompt",
) -> str:
    agent_profile = classifier.agent_profile
    app_description = agent_profile.description if agent_profile else None
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
        subject=prompt.subject,
        label=json.dumps(
            {
                "name": classification.label,
                "description": description,
                "passages": [item.quote for item in classification.evidence],
            },
            ensure_ascii=False,
        ),
        content=prompt.content,
        app_description=app_description,
        app_note=(
            classifier._get_prompt("review_app_note") if app_description else ""
        ),
    )
