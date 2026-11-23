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

DEFAULT_LABELS: List[Label] = [
    Label(
        "matches_user",
        "The response is written in the same language as the user's input.",
    ),
    Label(
        "mismatch",
        "The response is written in a different language from the user's input.",
    ),
]


class ResponseLanguageClassifier(Classifier):
    """Labels whether the assistant replied in the same language the user wrote in."""

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
        *,
        max_content_chars: int = MAX_CONTENT_CHARS,
        evidence_config: Optional[EvidenceConfig] = None,
        review_config: Optional[ReviewConfig] = None,
        agent_profile: Optional[AgentProfile] = None,
    ):
        super().__init__(
            name="response_language",
            labels=list(DEFAULT_LABELS),
            model=model,
            system_one_model=system_one_model,
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
