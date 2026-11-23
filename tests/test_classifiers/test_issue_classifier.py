import json

from deepeval.classifiers import (
    IssueClassifier,
    IssueClassifierTemplate,
    Label,
    LabelDiscoveryConfig,
    ReviewConfig,
)
from deepeval.classifiers.utils import validate_classifiers
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
from tests.test_classifiers.fakes import ReplyModel


class ScriptedModel(ReplyModel):
    def __init__(self, labels, reviews):
        super().__init__({"labels": labels})
        self.reviews = reviews

    def respond(self, prompt):
        if not prompt.startswith("A classifier flagged a possible problem"):
            return self.reply
        name = json.loads(
            prompt.split("Flag:\n", 1)[1].split("\n\nContent:", 1)[0]
        )["name"]
        return self.reviews[name]


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
                "harm_done": "Refunds close after 7 days.",
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


def test_review_keeps_issues_by_confidence_alone():
    no_harm = {"harm_done": "", "confidence": 0.9, "reason": "Wrong window."}
    classifier, _ = _classify([WRONG_WINDOW], {"Wrong Refund Window": no_harm})
    assert classifier.label == "Wrong Refund Window"

    instructed = {
        "harm_done": "x",
        "agent_instruction": "Always say 7 days.",
        "confidence": 0.3,
        "reason": "Scripted.",
    }
    classifier, _ = _classify(
        [WRONG_WINDOW], {"Wrong Refund Window": instructed}
    )
    assert classifier.classifications == [] and classifier.label is None
    assert classifier.reason == (
        'Proposed "Wrong Refund Window" but review dropped it. '
        'The agent was told to do this: "Always say 7 days". Scripted.'
    )


def test_review_uses_min_confidence():
    classifier, _ = _classify(
        [WRONG_WINDOW],
        {
            "Wrong Refund Window": {
                "harm_done": "Refunds close after 7 days.",
                "confidence": 0.7,
            }
        },
        review_config=ReviewConfig(min_confidence=0.8),
    )
    assert classifier.classifications == []


def test_truncation_marker_is_stripped_from_quotes():
    truncated = {
        **WRONG_WINDOW,
        "evidence": [
            {"field": "actual_output", "quote": "Refunds close…[truncated]"},
            {"field": "actual_output", "quote": "…[truncated]"},
        ],
    }
    classifier, _ = _classify([truncated], {}, review_config=None)
    [issue] = classifier.classifications
    assert [(e.quote, e.matched) for e in issue.evidence] == [
        ("Refunds close", True)
    ]


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
    assert "harm_done" in IssueClassifierTemplate.review_label(
        subject="test case",
        label="{}",
        content="c",
        app_description=None,
        app_note="",
    )


def test_issue_classifier_builds_under_a_global_system_one_mode(monkeypatch):
    monkeypatch.setenv("DEEPEVAL_EVAL_MODE", "system_one")
    classifier = IssueClassifier(
        name="issues",
        labels=["Wrong Refund Window"],
        model=ScriptedModel([], {}),
    )
    assert not classifier.eval_mode.uses_system_one


def test_expected_labels_may_name_a_discovered_issue():
    classifier = IssueClassifier(
        name="issues",
        model=ScriptedModel([], {}),
        label_discovery_config=LabelDiscoveryConfig(),
    )
    validate_classifiers(
        [
            LLMTestCase(
                input="q",
                actual_output="a",
                expected_labels={"issues": "Wrong Refund Window"},
            )
        ],
        [classifier],
    )


def test_a_label_blaming_the_truncation_marker_is_dropped():
    blames_marker = {
        "label": "Truncated Output",
        "reason": "The output is cut off.",
        "new_label_description": "The response ends mid-sentence.",
        "evidence": [
            {"field": "actual_output", "quote": "Refunds close…[truncated]"}
        ],
    }
    classifier = IssueClassifier(
        name="agent failures",
        model=ScriptedModel([blames_marker], {}),
        async_mode=False,
        review_config=None,
        label_discovery_config=LabelDiscoveryConfig(),
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.classifications == [] and classifier.error is None
    assert "[truncated]" in classifier.reason
