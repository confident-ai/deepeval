from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "resolved",
        "By the end of the interaction the user's goal is fully achieved.",
    ),
    Label(
        "unresolved",
        "By the end of the interaction the user's goal is not achieved and has not been handed over.",
    ),
    Label(
        "handed_over",
        "The interaction ends with the user handed over to a human or another channel instead of being resolved.",
    ),
]


class ResolutionClassifier(Classifier):
    """Labels whether an interaction, single reply or whole conversation, reached its expected end state."""

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        super().__init__(
            name="resolution",
            labels=list(DEFAULT_LABELS),
            model=model,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
