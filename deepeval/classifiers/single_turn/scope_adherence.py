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
            "in_scope",
            f"The request is within {domain} and the response addresses it.",
        ),
        Label(
            "deflected",
            f"The request falls outside {domain} and the response declines or redirects without answering it.",
        ),
        Label(
            "out_of_scope_answered",
            f"The request falls outside {domain} but the response answers it anyway.",
        ),
    ]


class ScopeAdherenceClassifier(Classifier):
    """Labels whether the assistant stayed inside its intended domain, deflected, or answered off-scope."""

    def __init__(
        self,
        scope: Optional[str] = None,
        name: str = "scope_adherence",
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
