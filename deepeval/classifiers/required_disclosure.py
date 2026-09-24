from typing import List, Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.config.eval_mode import ClassifierEvalModeName
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel


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
        self.disclosures = disclosures
        super().__init__(
            name="required_disclosure",
            labels=(
                labels
                if labels is not None
                else _default_labels(disclosures=disclosures)
            ),
            model=model,
            system_one_model=system_one_model,
            eval_mode=eval_mode,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
