import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Type, Union

from openai import APIError

from deepeval.classifiers.classifier.schema import (
    ClassificationResult,
    ClassificationResults,
    as_classification_results,
)
from deepeval.errors import DeepEvalError
from deepeval.metrics.utils.generation import (
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.test_run.api import Classification

if TYPE_CHECKING:
    from deepeval.classifiers.classifier.classifier import Classifier

MAX_GENERATION_ATTEMPTS = 2

# A failed provider call during review or label merging keeps the label
# rather than failing the whole classification.
PROVIDER_ERRORS = (APIError, asyncio.TimeoutError, TimeoutError)

OutputSchema = Type[Union[ClassificationResult, ClassificationResults]]


@dataclass(frozen=True)
class ClassificationStep:
    classifications: List[Classification]
    seconds: float


def generate_classification_results(
    classifier: "Classifier", prompt: str, output_schema: OutputSchema
) -> List[ClassificationResult]:
    """Asks the judge to classify, once more when its answer is not valid
    JSON for ``output_schema``."""
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            return generate_with_schema_and_extract(
                metric=classifier,
                prompt=prompt,
                schema_cls=output_schema,
                extract_schema=as_classification_results,
                extract_json=lambda data: as_classification_results(
                    output_schema.model_validate(data)
                ),
            )
        except (ValueError, DeepEvalError):
            if attempt == MAX_GENERATION_ATTEMPTS:
                raise
            classifier.error = None
    return []


async def a_generate_classification_results(
    classifier: "Classifier", prompt: str, output_schema: OutputSchema
) -> List[ClassificationResult]:
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            return await a_generate_with_schema_and_extract(
                metric=classifier,
                prompt=prompt,
                schema_cls=output_schema,
                extract_schema=as_classification_results,
                extract_json=lambda data: as_classification_results(
                    output_schema.model_validate(data)
                ),
            )
        except (ValueError, DeepEvalError):
            if attempt == MAX_GENERATION_ATTEMPTS:
                raise
            classifier.error = None
    return []
