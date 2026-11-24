from typing import Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import (
    AgentProfile,
    Classifier,
    ClassifierTemplate,
    EvidenceConfig,
    LabelDiscoveryConfig,
    ReviewConfig,
)
from deepeval.classifiers.classifier.steps.truncate import MAX_CONTENT_CHARS
from deepeval.classifiers.issues.schema import IssueReview
from deepeval.config.eval_mode import ClassifierEvalModeName
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSystemOneModel
from deepeval.templates import make_template_class
from deepeval.classifiers.utils import accepts_system_one_model


class IssueClassifierTemplate(
    make_template_class("IssueClassifier", feature="classifiers"),
    ClassifierTemplate,
):
    pass


class IssueClassifier(Classifier):
    """Finds the problems an AI agent caused in a test case, conversation or
    trace: every issue that applies, each with cited evidence, verified by an
    issue review that scores whether real, uninstructed harm happened."""

    default_template: Type[ClassifierTemplate] = IssueClassifierTemplate
    review_schema: Type[IssueReview] = IssueReview
    finds_issues: bool = True

    @accepts_system_one_model
    def __init__(
        self,
        name: str,
        labels: Sequence[Union[str, Label]] = (),
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        decision_model: Optional[Union[str, DeepEvalBaseSystemOneModel]] = None,
        eval_mode: Optional[ClassifierEvalModeName] = "llm",
        include_reason: bool = True,
        async_mode: bool = True,
        classification_template: Optional[Type[ClassifierTemplate]] = None,
        *,
        description: Optional[str] = None,
        max_content_chars: int = MAX_CONTENT_CHARS,
        evidence_config: Optional[EvidenceConfig] = EvidenceConfig(),
        review_config: Optional[ReviewConfig] = ReviewConfig(),
        label_discovery_config: Optional[LabelDiscoveryConfig] = None,
        agent_profile: Optional[AgentProfile] = None,
    ):
        super().__init__(
            name=name,
            labels=labels,
            model=model,
            decision_model=decision_model,
            eval_mode=eval_mode,
            include_reason=include_reason,
            allow_none=True,
            async_mode=async_mode,
            classification_template=(
                classification_template or self.default_template
            ),
            description=description,
            multi_label=True,
            max_content_chars=max_content_chars,
            evidence_config=evidence_config,
            review_config=review_config,
            label_discovery_config=label_discovery_config,
            agent_profile=agent_profile,
        )
