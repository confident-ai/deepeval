from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM

DEFAULT_LABELS: List[Label] = [
    Label(
        "escalated",
        "The response hands the user over to a human agent or another escalation path.",
    ),
    Label(
        "offered",
        "The response offers to escalate or connect the user with a human but does not do so yet.",
    ),
    Label(
        "not_escalated",
        "The response handles the request itself with no mention of escalation.",
    ),
]


class EscalationClassifier(Classifier):
    """Labels whether the assistant handed the user to a human, offered to, or handled it alone."""

    def __init__(
        self,
        name: str = "escalation",
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
