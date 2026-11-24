import asyncio
import json
import logging
from typing import TYPE_CHECKING, List, Optional

from deepeval.classifiers.classifier.schema import LabelReview
from deepeval.classifiers.classifier.steps.classification import (
    MAX_GENERATION_ATTEMPTS,
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


def review_classifications(
    classifier: "Classifier", prompt: "ClassificationPrompt"
) -> None:
    reviews = [
        _review(classifier, _build_review_prompt(classifier, c, prompt))
        for c in classifier.classifications
    ]
    _apply_review_results(classifier, reviews)


async def a_review_classifications(
    classifier: "Classifier", prompt: "ClassificationPrompt"
) -> None:
    reviews = await asyncio.gather(
        *(
            _a_review(classifier, _build_review_prompt(classifier, c, prompt))
            for c in classifier.classifications
        )
    )
    _apply_review_results(classifier, reviews)


def _review(classifier: "Classifier", prompt: str) -> Optional[LabelReview]:
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
    return None


async def _a_review(
    classifier: "Classifier", prompt: str
) -> Optional[LabelReview]:
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
    return None


def _apply_review_results(
    classifier: "Classifier", reviews: List[Optional[LabelReview]]
) -> None:
    kept_classifications, dropped_classifications = [], []
    for classification, review in zip(classifier.classifications, reviews):
        if review is not None:
            classification.confidence = review.confidence
        if (
            review is not None
            and review.confidence >= classifier.review_config.min_confidence
        ):
            kept_classifications.append(classification)
        else:
            dropped_classifications.append((classification, review))
    kept_classifications.sort(key=lambda c: -c.confidence)
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
