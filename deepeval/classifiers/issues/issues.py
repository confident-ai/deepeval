import re
from typing import List, Optional, Sequence, Type, Union

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
from deepeval.classifiers.issues.schema import IssueReview
from deepeval.models import DeepEvalBaseLLM
from deepeval.templates import make_template_class
from deepeval.test_run.api import Evidence

_QUOTE_TAIL_RE = re.compile(r"""[\s"'\\]+$""")
_TRUNCATED = "[truncated]"


class IssueClassifierTemplate(
    make_template_class("IssueClassifier", feature="classifiers"),
    ClassifierTemplate,
):
    pass


class IssueClassifier(Classifier):
    """Finds the problems an AI agent caused in a test case, conversation or
    trace: every issue that applies, each with cited evidence, verified by an
    issue review that keeps only real, uninstructed harm."""

    def __init__(
        self,
        name: str,
        labels: Sequence[Union[str, Label]] = (),
        description: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        classification_template: Type[
            IssueClassifierTemplate
        ] = IssueClassifierTemplate,
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
            classification_template=classification_template,
            allow_none=True,
            multi_label=True,
            max_content_chars=max_content_chars,
            evidence_config=evidence_config,
            review_config=review_config,
            label_discovery_config=label_discovery_config,
            classification_context=classification_context,
        )

    def _guidance_prompt(self) -> str:
        return self._get_prompt("issue_guidance")

    def _review_schema(self) -> Type[IssueReview]:
        return IssueReview

    def _passes_review(self, review: IssueReview) -> bool:
        return (
            bool(review.harm.strip())
            and not review.instruction.strip()
            and review.confidence >= self.review_config.min_confidence
        )

    def _review_reason(self, review: IssueReview) -> str:
        instruction = review.instruction.strip().rstrip(".")
        if not instruction:
            return review.reason
        return " ".join(
            part
            for part in (
                f'The agent was told to do this: "{instruction}".',
                review.reason,
            )
            if part
        )

    def _rejects_evidence(self, evidence: List[Evidence]) -> bool:
        return bool(evidence) and all(
            _QUOTE_TAIL_RE.sub("", item.quote).endswith(_TRUNCATED)
            for item in evidence
        )
