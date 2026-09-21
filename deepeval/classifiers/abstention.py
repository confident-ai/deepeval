from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "abstained",
        "The provided context does not contain the answer and the response says so instead of answering.",
    ),
    Label(
        "answered",
        "The provided context contains the answer and the response gives it.",
    ),
    Label(
        "fabricated",
        "The provided context does not contain the answer but the response gives one anyway.",
    ),
]


class AbstentionClassifier(Classifier):
    """Labels whether the assistant abstained, answered from context, or fabricated when context lacked the answer."""

    def __init__(
        self,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        super().__init__(
            name="abstention",
            labels=list(DEFAULT_LABELS),
            model=model,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
