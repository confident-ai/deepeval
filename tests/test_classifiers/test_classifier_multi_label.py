import json

from deepeval.classifiers import Classifier, Label
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


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
    input="How much is Pro, and can you cancel my order?",
    actual_output="Pro is $99. I can't help with orders.",
)
LABELS = [Label("pricing", "Asks a price."), Label("cancellation", "Asks to cancel.")]


def _classify(reply, **kwargs):
    model = ReplyModel(reply)
    classifier = Classifier(
        name="topic", labels=LABELS, model=model, async_mode=False, multi_label=True, **kwargs
    )
    classifier.classify(TEST_CASE, _show_indicator=False)
    return classifier, model.prompts[0]


def test_every_matching_label_is_returned_in_order():
    classifier, prompt = _classify(
        {
            "labels": [
                {"label": "pricing", "reason": "Asked the Pro price."},
                {"label": "Cancellation", "reason": "Asked to cancel."},
                {"reason": "missing label is dropped"},
            ]
        }
    )
    assert [r.label for r in classifier.results] == ["pricing", "cancellation"]
    assert classifier.label == "pricing" and classifier.reason == "Asked the Pro price."
    assert '"labels"' in prompt and "every label that applies" in prompt
    assert classifier.is_successful("cancellation")


def test_repeated_labels_merge_their_evidence():
    classifier, _ = _classify(
        {
            "labels": [
                {"label": "pricing", "reason": "a", "evidence": [{"field": "input", "quote": "How much is Pro"}]},
                {"label": "pricing", "reason": "b", "evidence": [{"field": "actual_output", "quote": "Pro is $99."}]},
            ]
        },
        include_evidence=True,
    )
    [result] = classifier.results
    assert [e.matched for e in result.evidence] == [True, True]


def test_none_item_gives_no_labels_and_its_reason():
    classifier, _ = _classify(
        {"labels": [{"label": "NONE", "reason": "Just a greeting."}]}, allow_none=True
    )
    assert classifier.results == [] and classifier.label is None
    assert classifier.reason == "Just a greeting." and classifier.error is None


def test_only_unknown_labels_is_an_error():
    classifier, _ = _classify({"labels": [{"label": "billing", "reason": "x"}]})
    assert classifier.results == [] and classifier.error is not None


def test_single_object_reply_is_accepted():
    classifier, _ = _classify({"label": "pricing", "reason": "Asked a price."})
    assert [r.label for r in classifier.results] == ["pricing"]


def test_multi_label_with_new_labels():
    classifier, _ = _classify(
        {
            "labels": [
                {"label": "pricing", "reason": "Price.", "new_label_description": None},
                {"label": "Order Help", "reason": "Order.", "new_label_description": "Cases about orders."},
            ]
        },
        suggest_labels=True,
    )
    assert [(r.label, r.new_label_description) for r in classifier.results] == [
        ("pricing", None),
        ("Order Help", "Cases about orders."),
    ]


def test_single_label_mode_fills_results():
    model = ReplyModel({"label": "pricing", "reason": "Price."})
    classifier = Classifier(name="topic", labels=LABELS, model=model, async_mode=False)
    classifier.classify(TEST_CASE, _show_indicator=False)
    assert [r.label for r in classifier.results] == ["pricing"]
    assert "labels\"" not in model.prompts[0]
