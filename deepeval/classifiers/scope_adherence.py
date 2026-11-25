from typing import List, Optional, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import (
    AgentProfile,
    Classifier,
    ClassifierTemplate,
    EvidenceConfig,
    ReviewConfig,
)
from deepeval.classifiers.classifier.steps.truncate import MAX_CONTENT_CHARS
from deepeval.config.eval_mode import ClassifierEvalModeName
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.classifiers.utils import accepts_system_one_model


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

    @accepts_system_one_model
    def __init__(
        self,
        scope: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        decision_model: Optional[Union[str, DeepEvalBaseSystemOneModel]] = None,
        eval_mode: Optional[ClassifierEvalModeName] = None,
        include_reason: bool = True,
        allow_none: bool = False,
        async_mode: bool = True,
        classification_template: Type[ClassifierTemplate] = ClassifierTemplate,
        *,
        max_content_chars: int = MAX_CONTENT_CHARS,
        evidence_config: Optional[EvidenceConfig] = None,
        review_config: Optional[ReviewConfig] = None,
        agent_profile: Optional[AgentProfile] = None,
    ):
        self.scope = scope
        super().__init__(
            name="scope_adherence",
            labels=_default_labels(scope=scope),
            model=model,
            decision_model=decision_model,
            eval_mode=eval_mode,
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=classification_template,
            max_content_chars=max_content_chars,
            evidence_config=evidence_config,
            review_config=review_config,
            agent_profile=agent_profile,
        )
