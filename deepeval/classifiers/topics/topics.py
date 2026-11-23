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
from deepeval.models import DeepEvalBaseLLM
from deepeval.templates import make_template_class


class TopicClassifierTemplate(
    make_template_class("TopicClassifier", feature="classifiers"),
    ClassifierTemplate,
):
    pass


class TopicClassifier(Classifier):
    """Labels what an interaction is mainly about, reviewing each label so a
    side mention or a routine step is not mistaken for the subject."""

    default_template: Type[ClassifierTemplate] = TopicClassifierTemplate

    def __init__(
        self,
        name: str,
        labels: Sequence[Union[str, Label]] = (),
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        allow_none: bool = True,
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
            eval_mode="llm",
            include_reason=include_reason,
            allow_none=allow_none,
            async_mode=async_mode,
            classification_template=(
                classification_template or self.default_template
            ),
            description=description,
            max_content_chars=max_content_chars,
            evidence_config=evidence_config,
            review_config=review_config,
            label_discovery_config=label_discovery_config,
            agent_profile=agent_profile,
        )
