import json

from deepeval.classifiers import (
    IssueClassifier,
    IssueClassifierTemplate,
    Label,
    LabelDiscoveryConfig,
    ReviewConfig,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

CASE = LLMTestCase(
    input="What's the refund window?",
    actual_output="Refunds close after 7 days.",
    retrieval_context=["Refunds are accepted within 30 days of purchase."],
)
WRONG_WINDOW = {
    "label": "Wrong Refund Window",
    "reason": "The agent said 7 days but the policy says 30.",
    "new_label_description": None,
    "evidence": [
        {"field": "actual_output", "quote": "Refunds close after 7 days."}
    ],
}


class ScriptedModel(DeepEvalBaseLLM):
    def __init__(self, labels, reviews):
        self.labels = labels
        self.reviews = reviews
        self.prompts = []

    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        if prompt.startswith("A classifier flagged a possible problem"):
            name = json.loads(
                prompt.split("Flag:\n", 1)[1].split("\n\nContent:", 1)[0]
            )["name"]
            return json.dumps(self.reviews[name])
        return json.dumps({"labels": self.labels})

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "scripted"


def _classify(labels, reviews, **kwargs):
    model = ScriptedModel(labels, reviews)
    classifier = IssueClassifier(
        name="agent failures",
        labels=[
            Label(
                "Wrong Refund Window",
                "States a refund window that differs from the policy.",
            )
        ],
        model=model,
        async_mode=False,
        **kwargs,
    )
    classifier.classify(CASE, _show_indicator=False)
    return classifier, model


def test_issue_prompts_add_guidance_and_cite_evidence():
    classifier, model = _classify(
        [WRONG_WINDOW],
        {
            "Wrong Refund Window": {
                "harm": "Refunds close after 7 days.",
                "confidence": 0.9,
            }
        },
    )
    prompt = model.prompts[0]
    assert "This classifier flags failures on the agent's side only" in prompt
    assert '"labels"' in prompt and '"evidence"' in prompt
    [issue] = classifier.classifications
    assert issue.label == "Wrong Refund Window" and issue.confidence == 0.9
    assert issue.evidence[0].matched
    assert model.prompts[1].startswith(
        "A classifier flagged a possible problem"
    )


def test_review_drops_issues_without_harm_or_that_were_instructed():
    no_harm = {
        "harm": "",
        "confidence": 0.9,
        "reason": "The customer still got help.",
    }
    instructed = {
        "harm": "x",
        "instruction": "Always say 7 days.",
        "confidence": 0.9,
        "reason": "Scripted.",
    }
    for review, reason in [
        (no_harm, "The customer still got help."),
        (
            instructed,
            'The agent was told to do this: "Always say 7 days". Scripted.',
        ),
    ]:
        classifier, _ = _classify(
            [WRONG_WINDOW], {"Wrong Refund Window": review}
        )
        assert classifier.classifications == [] and classifier.label is None
        assert (
            classifier.reason
            == f'Proposed "Wrong Refund Window" but review dropped it. {reason}'
        )


def test_review_uses_min_confidence():
    classifier, _ = _classify(
        [WRONG_WINDOW],
        {
            "Wrong Refund Window": {
                "harm": "Refunds close after 7 days.",
                "confidence": 0.7,
            }
        },
        review_config=ReviewConfig(min_confidence=0.8),
    )
    assert classifier.classifications == []


def test_issues_citing_only_the_truncation_marker_are_dropped():
    truncated = {
        **WRONG_WINDOW,
        "evidence": [
            {"field": "actual_output", "quote": "Refunds close…[truncated]"}
        ],
    }
    classifier, model = _classify([truncated], {}, review_config=None)
    assert classifier.classifications == [] and classifier.error is None
    assert "[truncated]" in classifier.reason


def test_discovered_issue_labels_use_the_issue_naming_rule_and_cap():
    new = [
        {
            "label": f"New Issue {i}",
            "reason": "r",
            "new_label_description": "d",
            "evidence": [],
        }
        for i in range(4)
    ]
    classifier, model = _classify(
        new,
        {},
        review_config=None,
        evidence_config=None,
        label_discovery_config=LabelDiscoveryConfig(max_new_labels=2),
    )
    assert "3-7 words that say exactly what happened" in model.prompts[0]
    assert [c.label for c in classifier.classifications] == [
        "New Issue 0",
        "New Issue 1",
    ]


def test_issue_classifier_can_start_without_labels():
    model = ScriptedModel([], {})
    classifier = IssueClassifier(
        name="agent failures",
        model=model,
        async_mode=False,
        label_discovery_config=LabelDiscoveryConfig(),
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.labels == [] and classifier.classifications == []


def test_template_overrides_only_the_issue_parts():
    assert IssueClassifierTemplate.classify_single_turn is not None
    assert "harm" in IssueClassifierTemplate.review_label(
        subject="test case",
        label="{}",
        content="c",
        app_description=None,
        app_note="",
    )
