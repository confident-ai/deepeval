from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "complete",
        "Every part of the user's request is addressed in the response.",
    ),
    Label(
        "partial",
        "Some parts of the user's request are addressed and others are not.",
    ),
    Label(
        "ignored",
        "None of the parts of the user's request are addressed.",
    ),
]


class InstructionCompletenessClassifier(Classifier):
    """Labels whether every part of a multi-part request was addressed."""

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        super().__init__(
            name="instruction_completeness",
            labels=list(DEFAULT_LABELS),
            model=model,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
