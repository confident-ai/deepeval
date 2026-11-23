from typing import Optional, Sequence, Type, Union

from deepeval.classifiers.base_classifier import Label
from deepeval.classifiers.classifier import (
    ClassificationContext,
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
        description: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        classification_template: Optional[Type[ClassifierTemplate]] = None,
        allow_none: bool = True,
        max_content_chars: int = MAX_CONTENT_CHARS,
        evidence_config: Optional[EvidenceConfig] = EvidenceConfig(),
        review_config: Optional[ReviewConfig] = ReviewConfig(),
        label_discovery_config: Optional[LabelDiscoveryConfig] = None,
        classification_context: Optional[ClassificationContext] = None,
    ):
        super().__init__(
            name=name,
            labels=labels,
            description=description,
            model=model,
            include_reason=include_reason,
            async_mode=async_mode,
            classification_template=(
                classification_template or self.default_template
            ),
            allow_none=allow_none,
            max_content_chars=max_content_chars,
            evidence_config=evidence_config,
            review_config=review_config,
            label_discovery_config=label_discovery_config,
            classification_context=classification_context,
        )
