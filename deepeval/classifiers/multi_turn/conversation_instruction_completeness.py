from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "complete",
        "Every request the user made during the conversation is addressed by the end.",
    ),
    Label(
        "partial",
        "Some of the user's requests are addressed by the end and others are not.",
    ),
    Label(
        "dropped",
        "Requests the user made earlier in the conversation are forgotten and never addressed.",
    ),
]


class ConversationInstructionCompletenessClassifier(Classifier):
    """Labels whether every part of a compound task spread across a conversation was eventually addressed."""

    def __init__(
        self,
        name: str = "conversation_instruction_completeness",
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
