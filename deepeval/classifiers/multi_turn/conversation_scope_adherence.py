from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM


def _default_labels(scope: Optional[str]) -> List[Label]:
    domain = (
        f"the application's intended scope ({scope})"
        if scope
        else "the application's intended scope"
    )
    return [
        Label(
            "stayed_in_scope", f"Every assistant turn stays within {domain}."
        ),
        Label(
            "drifted_and_recovered",
            f"At least one assistant turn leaves {domain} but a later turn steers the conversation back.",
        ),
        Label(
            "drifted",
            f"At least one assistant turn leaves {domain} and the conversation does not return to it.",
        ),
    ]


class ConversationScopeAdherenceClassifier(Classifier):
    """Labels whether the assistant stayed in scope across a whole conversation, drifted and recovered, or drifted."""

    def __init__(
        self,
        scope: Optional[str] = None,
        name: str = "conversation_scope_adherence",
        labels: Optional[Sequence[Union[str, Label]]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        evaluation_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        self.scope = scope
        super().__init__(
            name=name,
            labels=(
                labels if labels is not None else _default_labels(scope=scope)
            ),
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
            evaluation_template=evaluation_template,
        )
