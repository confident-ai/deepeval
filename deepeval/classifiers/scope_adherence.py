from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import Classifier, ClassifierTemplate
from deepeval.config.eval_mode import ClassifierEvalModeName
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel


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
        self.scope = scope
        super().__init__(
            name="scope_adherence",
            labels=_default_labels(scope=scope),
            model=model,
            system_one_model=system_one_model,
            eval_mode=eval_mode,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
        )
