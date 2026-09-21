from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.models import DeepEvalBaseLLM


def _default_labels(disclosures: Optional[Sequence[str]]) -> List[Label]:
    if disclosures:
        required = "the required elements: " + "; ".join(disclosures)
    else:
        required = "the disclosures the application is required to include, such as disclaimers, AI disclosure, or citations"
    return [
        Label("present", f"The response includes all of {required}."),
        Label("missing", f"The response includes none of {required}."),
        Label(
            "partial", f"The response includes some but not all of {required}."
        ),
    ]


class RequiredDisclosureClassifier(Classifier):
    """Labels whether mandated elements such as disclaimers, AI disclosure, or citations are present."""

    def __init__(
        self,
        disclosures: Optional[Sequence[str]] = None,
        name: str = "required_disclosure",
        labels: Optional[Sequence[Union[str, Label]]] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        evaluation_template: Type[ClassifierTemplate] = ClassifierTemplate,
    ):
        self.disclosures = disclosures
        super().__init__(
            name=name,
            labels=(
                labels
                if labels is not None
                else _default_labels(disclosures=disclosures)
            ),
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
            evaluation_template=evaluation_template,
        )
