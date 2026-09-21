from typing import List, Optional, Sequence, Type, Union

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
        name: str = "abstention",
        labels: Optional[Sequence[Union[str, Label]]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        evaluation_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        super().__init__(
            name=name,
            labels=labels if labels is not None else list(DEFAULT_LABELS),
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
            evaluation_template=evaluation_template,
        )
