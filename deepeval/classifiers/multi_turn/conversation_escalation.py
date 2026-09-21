from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "escalated",
        "At some point in the conversation the assistant hands the user over to a human or another escalation path.",
    ),
    Label(
        "offered",
        "The assistant offers to escalate at some point in the conversation but never actually does.",
    ),
    Label(
        "not_escalated",
        "The assistant never escalates or offers to escalate during the conversation.",
    ),
]


class ConversationEscalationClassifier(Classifier):
    """Labels whether, across a whole conversation, the assistant escalated at the right point."""

    def __init__(
        self,
        name: str = "conversation_escalation",
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
