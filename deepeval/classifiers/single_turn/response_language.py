from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "matches_user",
        "The response is written in the same language as the user's input.",
    ),
    Label(
        "mismatch",
        "The response is written in a different language from the user's input.",
    ),
]


class ResponseLanguageClassifier(Classifier):
    """Labels whether the assistant replied in the same language the user wrote in."""

    def __init__(
        self,
        name: str = "response_language",
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
