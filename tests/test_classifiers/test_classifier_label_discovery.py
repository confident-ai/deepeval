import pytest

from deepeval.classifiers import (
    Classifier,
    Label,
    LabelDiscoveryConfig,
    ReviewConfig,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from tests.test_classifiers.fakes import ReplyModel

TEST_CASE = LLMTestCase(
    input="Can I upgrade my plan?", actual_output="Yes, here's how."
)
LABELS = [Label("pricing", "Asks what something costs.")]


def _classify(reply, candidate_labels=(), **kwargs):
    model = ReplyModel(reply)
    classifier = Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        async_mode=False,
        label_discovery_config=LabelDiscoveryConfig(
            candidate_labels=list(candidate_labels)
        ),
        **kwargs,
    )
    classifier.classify(TEST_CASE, _show_indicator=False)
    return classifier, model.prompts[0]


def test_existing_label_has_no_description():
    classifier, prompt = _classify(
        {
            "label": "Pricing",
            "reason": "Asked a price.",
            "new_label_description": None,
        }
    )
    assert (
        classifier.label == "pricing"
        and classifier.classifications[0].new_label_description is None
    )
    assert (
        '"new_label_description"' in prompt and "Candidate Labels" not in prompt
    )


def test_new_label_keeps_the_judges_description():
    classifier, _ = _classify(
        {
            "label": "Plan Upgrade",
            "reason": "User asked to upgrade.",
            "new_label_description": "Cases where the user asks to change plan.",
        }
    )
    assert classifier.label == "Plan Upgrade" and classifier.error is None
    assert (
        classifier.classifications[0].new_label_description
        == "Cases where the user asks to change plan."
    )


@pytest.mark.parametrize("description", [None, "", "null", " None "])
def test_new_label_without_description_gets_a_fallback(description):
    classifier, _ = _classify(
        {
            "label": "Plan Upgrade",
            "reason": "User asked to upgrade.",
            "new_label_description": description,
        }
    )
    assert classifier.classifications[0].new_label_description == (
        "Test cases where user asked to upgrade. "
        "Does not apply when supporting evidence is absent."
    )


def test_candidate_label_is_reused_with_its_description():
    classifier, prompt = _classify(
        {
            "label": "plan upgrade",
            "reason": "Upgrade.",
            "new_label_description": "x",
        },
        candidate_labels=[Label("Plan Upgrade", "Asks to change plan.")],
    )
    assert classifier.label == "Plan Upgrade"
    assert (
        classifier.classifications[0].new_label_description
        == "Asks to change plan."
    )
    assert "Candidate Labels:\n- Plan Upgrade: Asks to change plan." in prompt


@pytest.mark.parametrize("name", ["NONE", "No Topic Found", "Other"])
def test_absence_names_are_none(name):
    classifier, _ = _classify(
        {
            "label": name,
            "reason": "Just a greeting.",
            "new_label_description": "x",
        },
        allow_none=True,
    )
    assert classifier.label is None and classifier.error is None


def test_suggestion_is_rejected_in_system_one_mode():
    with pytest.raises(ValueError):
        Classifier(
            name="topic",
            labels=LABELS,
            model=ReplyModel({}),
            eval_mode="decision_model",
            label_discovery_config=LabelDiscoveryConfig(),
        )


def test_conversation_prompt_includes_suggest_labels_rules():
    model = ReplyModel(
        {"label": "Plan Upgrade", "reason": "r", "new_label_description": "d"}
    )
    Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        async_mode=False,
        label_discovery_config=LabelDiscoveryConfig(),
    ).classify(
        ConversationalTestCase(turns=[Turn(role="user", content="upgrade?")]),
        _show_indicator=False,
    )
    assert "propose a NEW label" in model.prompts[0]


def _separately(label_reply, discovery_reply, **kwargs):
    labeler, discoverer = ReplyModel(label_reply), ReplyModel(discovery_reply)
    classifier = Classifier(
        name="topics",
        labels=LABELS,
        model=labeler,
        async_mode=kwargs.pop("async_mode", False),
        label_discovery_config=LabelDiscoveryConfig(model=discoverer),
        **kwargs,
    )
    classifier.classify(TEST_CASE, _show_indicator=False)
    return classifier, labeler, discoverer


@pytest.mark.parametrize("async_mode", [True, False])
def test_a_separate_discovery_model_is_not_asked_when_a_label_fits(async_mode):
    classifier, labeler, discoverer = _separately(
        {"label": "pricing", "reason": "asks a price"},
        {"label": "Plan Upgrade", "reason": "r"},
        async_mode=async_mode,
    )
    assert classifier.label == "pricing"
    assert discoverer.prompts == []
    assert "new_label_description" not in labeler.prompts[0]
    assert classifier.steps.suggestion is None


@pytest.mark.parametrize("async_mode", [True, False])
def test_a_separate_discovery_model_proposes_when_nothing_fits(async_mode):
    classifier, _, discoverer = _separately(
        {"label": "NONE", "reason": "not a price"},
        {
            "label": "Plan Upgrade",
            "reason": "asks to upgrade",
            "new_label_description": "Asks to move to a higher plan.",
        },
        async_mode=async_mode,
    )
    assert classifier.label == "Plan Upgrade"
    assert "pricing" not in discoverer.prompts[0]
    assert [label.name for label in classifier.candidate_labels] == [
        "Plan Upgrade"
    ]
    assert classifier.steps.classification is not None
    assert classifier.steps.suggestion is not None


def test_a_separate_discovery_model_cannot_pick_a_declared_label():
    classifier, _, _ = _separately(
        {"label": "NONE", "reason": "not a price"},
        {"label": "Pricing", "reason": "r", "new_label_description": "d"},
        allow_none=True,
    )
    assert classifier.label is None and classifier.classifications == []


def test_discovered_labels_are_reviewed_by_the_classifier_model():
    classifier, labeler, discoverer = _separately(
        {"label": "NONE", "reason": "r", "confidence": 0.9},
        {
            "label": "Plan Upgrade",
            "reason": "asks to upgrade",
            "new_label_description": "Asks to move to a higher plan.",
        },
        review_config=ReviewConfig(),
    )
    assert classifier.label == "Plan Upgrade"
    assert any("Plan Upgrade" in prompt for prompt in labeler.prompts[1:])
    assert len(discoverer.prompts) == 1
    assert classifier.steps.review is not None
