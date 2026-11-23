import json

import pytest

from deepeval.classifiers import (
    ClassificationContext,
    Classifier,
    EvidenceConfig,
    Label,
    RefusalClassifier,
    ReviewConfig,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class ReviewingModel(DeepEvalBaseLLM):
    def __init__(self, reply, confidences):
        self.reply = reply
        self.confidences = confidences
        self.prompts = []

    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        if prompt.startswith("A classifier labelled"):
            label = json.loads(
                prompt.split("Label:\n", 1)[1].split("\n\nContent:", 1)[0]
            )
            confidence = self.confidences[label["name"]]
            if confidence is None:
                return "not json"
            return json.dumps(
                {
                    "summary": "s",
                    "confidence": confidence,
                    "reason": f"{label['name']} checked.",
                }
            )
        return json.dumps(self.reply)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "reviewing"


CASE = LLMTestCase(
    input="How much is Pro, and can I cancel?", actual_output="Pro is $20."
)
LABELS = [
    Label("pricing", "Asks a price."),
    Label("cancellation", "Asks to cancel."),
]
BOTH = {
    "labels": [
        {"label": "cancellation", "reason": "Asked to cancel."},
        {"label": "pricing", "reason": "Asked a price."},
    ]
}


@pytest.mark.parametrize("async_mode", [True, False])
def test_each_label_gets_a_confidence_and_is_sorted_by_it(async_mode):
    model = ReviewingModel(BOTH, {"pricing": 0.9, "cancellation": 0.7})
    classifier = Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        multi_label=True,
        review_config=ReviewConfig(),
        async_mode=async_mode,
    )
    classifier.classify(CASE, _show_indicator=False)
    assert [(r.label, r.confidence) for r in classifier.classifications] == [
        ("pricing", 0.9),
        ("cancellation", 0.7),
    ]
    assert classifier.label == "pricing"


def test_labels_below_min_confidence_are_dropped():
    model = ReviewingModel(BOTH, {"pricing": 0.9, "cancellation": 0.3})
    classifier = Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        multi_label=True,
        review_config=ReviewConfig(),
        async_mode=False,
    )
    classifier.classify(CASE, _show_indicator=False)
    assert [r.label for r in classifier.classifications] == ["pricing"]


def test_all_dropped_explains_why():
    model = ReviewingModel(
        {"label": "pricing", "reason": "r"}, {"pricing": 0.2}
    )
    classifier = Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        review_config=ReviewConfig(min_confidence=0.5),
        async_mode=False,
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label is None and classifier.error is None
    assert (
        classifier.reason
        == 'Proposed "pricing" but review dropped it. pricing checked.'
    )


def test_unreadable_review_keeps_the_label_without_confidence():
    model = ReviewingModel(
        {"label": "pricing", "reason": "r"}, {"pricing": None}
    )
    classifier = Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        review_config=ReviewConfig(),
        async_mode=False,
    )
    classifier.classify(CASE, _show_indicator=False)
    assert (
        classifier.label == "pricing"
        and classifier.classifications[0].confidence is None
    )


def test_review_sees_label_description_passages_and_content():
    model = ReviewingModel(
        {
            "label": "pricing",
            "reason": "r",
            "evidence": [{"field": "input", "quote": "How much is Pro"}],
        },
        {"pricing": 0.8},
    )
    Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        review_config=ReviewConfig(),
        evidence_config=EvidenceConfig(),
        async_mode=False,
    ).classify(CASE, _show_indicator=False)
    review = model.prompts[-1]
    assert (
        '"description": "Asks a price."' in review
        and '"How much is Pro"' in review
    )
    assert "Pro is $20." in review and "test case" in review


def test_without_review_there_is_one_call_and_no_confidence():
    model = ReviewingModel({"label": "pricing", "reason": "r"}, {})
    classifier = Classifier(
        name="topic", labels=LABELS, model=model, async_mode=False
    )
    classifier.classify(CASE, _show_indicator=False)
    assert (
        len(model.prompts) == 1
        and classifier.classifications[0].confidence is None
    )


def test_description_and_context_reach_the_prompts():
    model = ReviewingModel(
        {"label": "pricing", "reason": "r"}, {"pricing": 0.8}
    )
    Classifier(
        name="topic",
        labels=LABELS,
        description="What the customer wants to talk about.",
        model=model,
        review_config=ReviewConfig(),
        classification_context=ClassificationContext(
            app_description="A billing assistant for a SaaS product.",
            trace_description="Each trace is one support chat.",
        ),
        async_mode=False,
    ).classify(CASE, _show_indicator=False)
    classify, review = model.prompts
    assert (
        'This classifier is named "topic". Its purpose: "What the customer wants to talk about."'
        in classify
    )
    assert (
        "The application being traced is: A billing assistant for a SaaS product."
        in classify
    )
    assert "Each trace is one support chat." in classify
    assert "App description:\nA billing assistant for a SaaS product." in review


def test_context_without_descriptions_asks_to_infer_the_use_case():
    model = ReviewingModel({"label": "pricing", "reason": "r"}, {})
    Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        classification_context=ClassificationContext(),
        async_mode=False,
    ).classify(CASE, _show_indicator=False)
    assert "Infer the application's use case" in model.prompts[0]
    assert "This classifier is named" not in model.prompts[0]


def test_built_in_classifiers_accept_evidence_and_review():
    model = ReviewingModel(
        {
            "label": "refused",
            "reason": "Declined the request.",
            "evidence": [
                {"field": "actual_output", "quote": "I can't help with that."}
            ],
        },
        {"refused": 0.85},
    )
    classifier = RefusalClassifier(
        model=model,
        evidence_config=EvidenceConfig(),
        review_config=ReviewConfig(),
        async_mode=False,
    )
    classifier.classify(
        LLMTestCase(
            input="How do I pick a lock?",
            actual_output="I can't help with that.",
        ),
        _show_indicator=False,
    )
    [classification] = classifier.classifications
    assert (
        classification.label == "refused" and classification.confidence == 0.85
    )
    assert classification.evidence[0].matched
    assert type(classifier.copy()) is RefusalClassifier
