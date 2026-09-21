from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "resolved",
        "By the end of the conversation the user's goal or scenario is fully achieved.",
    ),
    Label(
        "unresolved",
        "By the end of the conversation the user's goal or scenario is not achieved and has not been handed over.",
    ),
    Label(
        "handed_over",
        "The conversation ends with the user handed over to a human or another channel instead of being resolved.",
    ),
]


class ConversationResolutionClassifier(Classifier):
    """Labels whether a whole conversation reached its expected end state."""

    def __init__(
        self,
        name: str = "conversation_resolution",
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
