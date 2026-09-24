from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.config.eval_mode import ClassifierEvalModeName
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel

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
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        system_one_model: Optional[
            Union[str, DeepEvalBaseSystemOneModel]
        ] = None,
        eval_mode: Optional[ClassifierEvalModeName] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        super().__init__(
            name="escalation",
            labels=list(DEFAULT_LABELS),
            model=model,
            system_one_model=system_one_model,
            eval_mode=eval_mode,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
