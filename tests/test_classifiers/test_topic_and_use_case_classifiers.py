import pytest

from deepeval.classifiers import (
    AgentProfile,
    TopicClassifier,
    UseCaseClassifier,
)
from deepeval.test_case import LLMTestCase

CASE = LLMTestCase(
    input="How much is the Pro plan?", actual_output="Pro is $20 a month."
)
from tests.test_classifiers.fakes import ReplyModel


class ScriptedModel(ReplyModel):
    def __init__(self, confidence):
        super().__init__(
            {
                "label": "Pricing",
                "reason": "Asked a price.",
                "evidence": [
                    {"field": "input", "quote": "How much is the Pro plan?"}
                ],
            }
        )
        self.confidence = confidence

    def respond(self, prompt):
        if prompt.startswith("A classifier tagged"):
            return {
                "summary": "s",
                "confidence": self.confidence,
                "reason": "r",
            }
        return self.reply


@pytest.mark.parametrize(
    "classifier_class, review_opening",
    [
        (TopicClassifier, "with a topic label"),
        (UseCaseClassifier, "with a use-case label"),
    ],
)
def test_review_uses_its_own_prompt_and_context_note(
    classifier_class, review_opening
):
    model = ScriptedModel(0.9)
    classifier = classifier_class(
        name="topics",
        labels=["Pricing", "Support"],
        model=model,
        async_mode=False,
        agent_profile=AgentProfile(description="A billing assistant."),
    )
    classifier.classify(CASE, _show_indicator=False)
    classify, review = model.prompts
    assert '"evidence"' in classify
    assert (
        review_opening in review
        and 'When an "App description" is given' in review
    )
    [classification] = classifier.classifications
    assert (
        classification.label == "Pricing" and classification.confidence == 0.9
    )
    assert classification.evidence[0].matched


def test_low_confidence_topic_is_dropped():
    classifier = TopicClassifier(
        name="topics",
        labels=["Pricing"],
        model=ScriptedModel(0.3),
        async_mode=False,
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.classifications == [] and classifier.label is None


def test_copies_keep_their_class_and_template():
    classifier = UseCaseClassifier(
        name="use cases", labels=["Pricing"], model=ScriptedModel(0.9)
    )
    clone = classifier.copy()
    assert type(clone) is UseCaseClassifier
    assert clone.classification_template is classifier.classification_template


def test_cache_tells_topic_and_use_case_classifiers_apart():
    from deepeval.test_run.cache import Cache

    configurations = [
        Cache.create_classifier_configuration(
            classifier_class(
                name="c", labels=["Pricing"], model=ScriptedModel(0.9)
            )
        )
        for classifier_class in (TopicClassifier, UseCaseClassifier)
    ]
    assert configurations[0] != configurations[1]
