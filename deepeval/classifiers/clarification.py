from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "asked_clarification",
        "The input is ambiguous or underspecified and the response asks a clarifying question before proceeding.",
    ),
    Label(
        "answered_directly",
        "The input is clear enough and the response answers or acts on it without needing clarification.",
    ),
    Label(
        "guessed",
        "The input is ambiguous or underspecified but the response proceeds on an assumption instead of asking.",
    ),
]


class ClarificationClassifier(Classifier):
    """Labels whether the assistant asked for clarification, answered directly, or guessed on ambiguous input."""

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        super().__init__(
            name="clarification",
            labels=list(DEFAULT_LABELS),
            model=model,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
