import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from deepeval.classifiers.classifier.schema import LabelReview
from deepeval.classifiers.classifier.steps.classification import (
    MAX_GENERATION_ATTEMPTS,
    PROVIDER_ERRORS,
)
from deepeval.errors import DeepEvalError
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

_NO_RESPONSE = "The reviewer did not respond."


@dataclass(frozen=True)
class ClassificationReview:
    label: str
    kept: bool
    confidence: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class ReviewStep:
    reviews: List[ClassificationReview]
    seconds: float


def review_classifications(
    classifier: "Classifier", prompt: "ClassificationPrompt"
) -> List[ClassificationReview]:
    reviews = [
        _review(classifier, _build_review_prompt(classifier, c, prompt))
        for c in classifier.classifications
    ]
    return _apply_review_results(classifier, reviews)


async def a_review_classifications(
    classifier: "Classifier", prompt: "ClassificationPrompt"
) -> List[ClassificationReview]:
    reviews = await asyncio.gather(
        *(
            _a_review(classifier, _build_review_prompt(classifier, c, prompt))
            for c in classifier.classifications
        )
    )
    return _apply_review_results(classifier, reviews)


def _review(
    classifier: "Classifier", prompt: str
) -> Union[LabelReview, str, None]:
    schema = classifier.review_schema
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return generate_with_schema_and_extract(
                metric=classifier,
                prompt=prompt,
                schema_cls=schema,
                extract_schema=lambda review: review,
                extract_json=schema.model_validate,
            )
        except (ValueError, DeepEvalError) as error:
            logger.warning(f"Unreadable review: {error}")
            classifier.error = None
        except PROVIDER_ERRORS as error:
            logger.warning(f"Review call failed; keeping the label: {error}")
            return _NO_RESPONSE
    return None


async def _a_review(
    classifier: "Classifier", prompt: str
) -> Union[LabelReview, str, None]:
    schema = classifier.review_schema
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return await a_generate_with_schema_and_extract(
                metric=classifier,
                prompt=prompt,
                schema_cls=schema,
                extract_schema=lambda review: review,
                extract_json=schema.model_validate,
            )
        except (ValueError, DeepEvalError) as error:
            logger.warning(f"Unreadable review: {error}")
            classifier.error = None
        except PROVIDER_ERRORS as error:
            logger.warning(f"Review call failed; keeping the label: {error}")
            return _NO_RESPONSE
    return None


def _apply_review_results(
    classifier: "Classifier", reviews: List[Union[LabelReview, str, None]]
) -> List[ClassificationReview]:
    kept_classifications, dropped_classifications = [], []
    classification_reviews = []
    for classification, review in zip(classifier.classifications, reviews):
        if review == _NO_RESPONSE:
            kept_classifications.append(classification)
            classification_reviews.append(
                ClassificationReview(
                    label=classification.label, kept=True, reason=_NO_RESPONSE
                )
            )
            continue
        if review is not None:
            classification.confidence = review.confidence
        kept = (
            review is not None
            and review.confidence >= classifier.review_config.min_confidence
        )
        if kept:
            kept_classifications.append(classification)
        else:
            dropped_classifications.append((classification, review))
        classification_reviews.append(
            ClassificationReview(
                label=classification.label,
                kept=kept,
                confidence=review.confidence if review is not None else None,
                reason=(
                    review.explanation()
                    if review is not None
                    else "The review could not be read."
                ),
            )
        )
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
            dropped_classifications,
            key=lambda pair: -1.0 if pair[1] is None else pair[1].confidence,
        )
        classifier.reason = (
            f'Proposed "{classification.label}" but its review could not be read.'
            if review is None
            else " ".join(
                part
                for part in (
                    f'Proposed "{classification.label}" but review dropped it.',
                    review.explanation(),
                )
                if part
            )
        )
    return classification_reviews


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
