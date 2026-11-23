import json

import pytest

from deepeval.classifiers import (
    Classifier,
    EvidenceConfig,
    Label,
    LabelDiscoveryConfig,
)
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


def _classify(test_case, evidence, **kwargs):
    model = ReplyModel(
        {"label": "pricing", "reason": "Asked a price.", "evidence": evidence}
    )
    classifier = Classifier(
        name="topic",
        labels=[Label("pricing", "Asks what something costs."), "support"],
        model=model,
        async_mode=False,
        evidence_config=EvidenceConfig(),
        **kwargs,
    )
    classifier.classify(test_case, _show_indicator=False)
    return classifier, model.prompts[0]


def test_test_case_quotes_are_checked_against_their_field():
    classifier, prompt = _classify(
        LLMTestCase(
            input="How much is the Pro plan?",
            actual_output="It is $20 a month.",
        ),
        [
            {
                "field": "input",
                "quote": "How much is the Pro plan?",
                "note": "Asks the price.",
            },
            {
                "field": "actual_output",
                "quote": "It is $99 a month.",
                "note": "Wrong.",
            },
            {"field": "input", "note": "no quote, dropped"},
        ],
    )
    assert classifier.label == "pricing"
    assert [
        (e.field, e.matched) for e in classifier.classifications[0].evidence
    ] == [
        ("input", True),
        ("actual_output", False),
    ]
    assert (
        'one of "input", "actual_output"' in prompt and '"evidence"' in prompt
    )


def test_trace_quotes_resolve_spans_by_index():
    test_case = LLMTestCase(input="How much is Pro?", actual_output="$20.")
    test_case._trace_dict = {
        "name": "agent",
        "type": "agent",
        "children": [
            {
                "name": "lookup",
                "type": "tool",
                "output": "price: 20 USD",
                "children": [],
            }
        ],
    }
    classifier, prompt = _classify(
        test_case,
        [
            {"field": "output", "quote": "price: 20 USD", "span_index": 2},
            {"field": "output", "quote": "price: 20 USD", "span_index": 1},
        ],
    )
    assert [e.matched for e in classifier.classifications[0].evidence] == [
        True,
        False,
    ]
    assert '"span_index"' in prompt


def test_conversation_quotes_resolve_turns_by_position():
    classifier, prompt = _classify(
        ConversationalTestCase(
            scenario="Customer compares plans.",
            turns=[
                Turn(role="user", content="What does Pro cost?"),
                Turn(role="assistant", content="Pro costs $20 per month."),
            ],
        ),
        [
            {
                "field": "content",
                "quote": "Pro costs $20 per month.",
                "turn_index": 2,
            },
            {"field": "scenario", "quote": "Customer compares plans."},
        ],
    )
    assert [e.matched for e in classifier.classifications[0].evidence] == [
        True,
        True,
    ]
    assert '"turn_index"' in prompt and '"scenario"' in prompt


def test_evidence_combines_with_new_labels():
    model = ReplyModel(
        {
            "label": "Plan Upgrade",
            "reason": "Asked to upgrade.",
            "new_label_description": "Cases where the user asks to change plan.",
            "evidence": [{"field": "input", "quote": "Can I upgrade my plan?"}],
        }
    )
    classifier = Classifier(
        name="topic",
        labels=["pricing"],
        model=model,
        async_mode=False,
        evidence_config=EvidenceConfig(),
        label_discovery_config=LabelDiscoveryConfig(),
    )
    classifier.classify(
        LLMTestCase(input="Can I upgrade my plan?"), _show_indicator=False
    )
    assert (
        classifier.label == "Plan Upgrade"
        and classifier.classifications[0].evidence[0].matched
    )


def test_none_has_no_evidence():
    model = ReplyModel(
        {
            "label": "NONE",
            "reason": "Greeting.",
            "evidence": [{"field": "input", "quote": "hello there"}],
        }
    )
    classifier = Classifier(
        name="topic",
        labels=["pricing"],
        model=model,
        async_mode=False,
        evidence_config=EvidenceConfig(),
        allow_none=True,
    )
    classifier.classify(LLMTestCase(input="hello there"), _show_indicator=False)
    assert classifier.label is None and classifier.classifications == []


def test_prompt_without_evidence_has_no_evidence_field():
    model = ReplyModel({"label": "pricing", "reason": "r"})
    Classifier(
        name="topic", labels=["pricing"], model=model, async_mode=False
    ).classify(LLMTestCase(input="price?"), _show_indicator=False)
    assert '"evidence"' not in model.prompts[0]


def test_evidence_is_rejected_in_system_one_mode():
    with pytest.raises(ValueError):
        Classifier(
            name="topic",
            labels=["pricing"],
            model=ReplyModel({}),
            eval_mode="system_one",
            evidence_config=EvidenceConfig(),
        )


def test_evidence_config_sets_the_limits():
    model = ReplyModel(
        {
            "label": "pricing",
            "reason": "r",
            "evidence": [
                {"field": "input", "quote": "How much is the Pro plan?"}
            ],
        }
    )
    classifier = Classifier(
        name="topic",
        labels=["pricing"],
        model=model,
        async_mode=False,
        evidence_config=EvidenceConfig(max_items=5, max_quote_chars=12),
    )
    classifier.classify(
        LLMTestCase(input="How much is the Pro plan?"), _show_indicator=False
    )
    assert (
        "Cite up to 5 evidence items" in model.prompts[0]
        and "at most 12 characters" in model.prompts[0]
    )
    assert (
        classifier.classifications[0].evidence[0].quote == "How much is "
        and classifier.classifications[0].evidence[0].matched
    )
