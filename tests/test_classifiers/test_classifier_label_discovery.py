import json
import pytest

from deepeval.classifiers import Classifier, Label, LabelDiscoveryConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn


class ReplyModel(DeepEvalBaseLLM):
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        return json.dumps(self.reply)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "reply-model"


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


def test_new_label_without_description_gets_a_fallback():
    classifier, _ = _classify(
        {
            "label": "Plan Upgrade",
            "reason": "User asked to upgrade.",
            "new_label_description": None,
        }
    )
    assert classifier.classifications[0].new_label_description.startswith(
        "Cases where user asked to upgrade"
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
            eval_mode="system_one",
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
