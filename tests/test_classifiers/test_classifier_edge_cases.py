import httpx
import openai
import pytest

from deepeval.classifiers import (
    ReviewConfig,
    Classifier,
    EvidenceConfig,
    Label,
    LabelDiscoveryConfig,
)
from deepeval.classifiers.classifier.schema import EvidenceItem
from deepeval.classifiers.classifier.steps.evidence import (
    blames_truncation_marker,
)
from deepeval.classifiers.utils import _is_absence_label
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from tests.test_classifiers.fakes import ReplyModel, WordEmbedding

CASE = LLMTestCase(input="How much is Pro?", actual_output="Pro is $20.")
LABELS = [Label("pricing", "Asks a price."), Label("other", "Anything else.")]


def _classify(reply, **kwargs):
    classifier = Classifier(
        name="t",
        labels=kwargs.pop("labels", LABELS),
        model=kwargs.pop("model", None) or ReplyModel(reply),
        async_mode=False,
        **kwargs,
    )
    classifier.classify(CASE, _show_indicator=False)
    return classifier


@pytest.mark.parametrize(
    "name",
    ["Other", "No Issues", "No Clear Issues", "No Issue Found", "None Found"],
)
def test_names_that_only_say_nothing_was_found_are_no_label(name):
    assert _is_absence_label(name, "Issues")


@pytest.mark.parametrize(
    "name",
    [
        "No Error Handling",
        "No Results Found",
        "No Escalation Offered For Billing Issues",
        "No Topic Found",
    ],
)
def test_other_names_starting_with_no_are_real_labels(name):
    assert not _is_absence_label(name, "Issues")


def test_absence_uses_the_classifier_name():
    assert _is_absence_label("No Use Case Found", "Use Cases")
    assert _is_absence_label("No Topic", "topics")


def test_marker_in_reason_alone_is_not_a_truncation_claim():
    reason = "Agent said 7 days; the tool output, cut at …[truncated], says 30."
    assert not blames_truncation_marker("policy_misquote", None, reason, [])


def test_truncation_label_with_real_evidence_is_kept():
    evidence = [
        EvidenceItem(field="actual_output", quote="The answer stops mid"),
        EvidenceItem(field="output", quote='tool says …[truncated]",'),
    ]
    assert not blames_truncation_marker("Answer Cut Off", None, "r", evidence)
    assert blames_truncation_marker("Answer Cut Off", None, "r", evidence[1:])


@pytest.mark.parametrize(
    "kwargs", [{}, {"multi_label": True}, {"evidence_config": EvidenceConfig()}]
)
def test_null_or_non_string_reason_is_tolerated(kwargs):
    for reason in (None, 5):
        classifier = _classify({"label": "pricing", "reason": reason}, **kwargs)
        assert classifier.label == "pricing" and classifier.error is None


def test_null_evidence_note_keeps_the_evidence():
    classifier = _classify(
        {
            "label": "pricing",
            "reason": "r",
            "evidence": [
                {"field": "input", "quote": "How much is Pro?", "note": None}
            ],
        },
        evidence_config=EvidenceConfig(),
    )
    assert classifier.classifications[0].evidence[0].matched


def test_wrong_or_missing_turn_index_is_corrected_to_the_turn_holding_the_quote():
    case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Can I get a refund?"),
            Turn(role="assistant", content="Refunds take 30 days."),
        ]
    )
    classifier = Classifier(
        name="t",
        labels=LABELS,
        model=ReplyModel(
            {
                "label": "other",
                "reason": "r",
                "evidence": [
                    {
                        "field": "content",
                        "quote": "Refunds take 30 days",
                        "turn_index": 9,
                    },
                    {"field": "content", "quote": "Refunds take 30 days"},
                ],
            }
        ),
        evidence_config=EvidenceConfig(),
        async_mode=False,
    )
    classifier.classify(case, _show_indicator=False)
    wrong, missing = classifier.classifications[0].evidence
    assert (wrong.matched, wrong.turn_index) == (True, 2)
    assert (missing.matched, missing.turn_index) == (True, 2)


def test_one_huge_turn_is_truncated_to_the_budget():
    case = ConversationalTestCase(
        turns=[Turn(role="user", content="x" * 50_000)], scenario="y" * 50_000
    )
    classifier = Classifier(
        name="t", labels=LABELS, model=ReplyModel(), max_content_chars=1000
    )
    assert len(classifier._build_prompt(case).content) < 1200


def test_tiny_budget_is_rejected():
    with pytest.raises(ValueError):
        Classifier(name="t", labels=LABELS, max_content_chars=3)


class ShortEmbedding(WordEmbedding):
    def embed_texts(self, texts):
        return super().embed_texts(texts)[:1]


class FailingEmbedding(WordEmbedding):
    def embed_texts(self, texts):
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://e"))


@pytest.mark.parametrize("embedding", [ShortEmbedding(), FailingEmbedding()])
def test_a_failed_embedding_falls_back_to_shared_words(embedding):
    classifier = _classify(
        {"label": "Refund Delay", "reason": "r", "new_label_description": "d"},
        labels=[Label("pricing", "Asks a price.")],
        label_discovery_config=LabelDiscoveryConfig(embedding_model=embedding),
    )
    assert classifier.label == "Refund Delay" and classifier.error is None


class FailingCallModel(ReplyModel):
    def __init__(self, reply, failing_prompt):
        super().__init__(reply)
        self.failing_prompt = failing_prompt

    def respond(self, prompt):
        if self.failing_prompt in prompt:
            raise openai.APITimeoutError(
                request=httpx.Request("POST", "http://m")
            )
        return self.reply


def test_a_review_the_provider_fails_keeps_the_label_unreviewed():
    classifier = Classifier(
        name="t",
        labels=LABELS,
        model=FailingCallModel(
            {"label": LABELS[0].name, "reason": "r"}, "Label:\n"
        ),
        review_config=ReviewConfig(),
        async_mode=False,
    )
    classifier.classify(CASE, _show_indicator=False)
    assert classifier.label == LABELS[0].name
    assert classifier.classifications[0].confidence is None
    [review] = classifier.steps.review.reviews
    assert review.kept and review.reason == "The reviewer did not respond."


def test_a_merge_judge_the_provider_fails_keeps_the_new_label():
    classifier = _classify(
        {"label": "Refund Delay", "reason": "r", "new_label_description": "d"},
        labels=[Label("Refund Request", "Asks for a refund.")],
        model=FailingCallModel(
            {
                "label": "Refund Delay",
                "reason": "r",
                "new_label_description": "d",
            },
            "newly proposed classifier label",
        ),
        label_discovery_config=LabelDiscoveryConfig(
            embedding_model=FailingEmbedding()
        ),
    )
    assert classifier.label == "Refund Delay" and classifier.error is None


def test_turn_metadata_reaches_the_judge():
    model = ReplyModel({"label": LABELS[0].name, "reason": "r"})
    classifier = Classifier(
        name="t", labels=LABELS, model=model, async_mode=False
    )
    classifier.classify(
        ConversationalTestCase(
            turns=[
                Turn(role="user", content="hi"),
                Turn(
                    role="assistant",
                    content="sorry",
                    metadata={"error": "Timeout"},
                ),
            ]
        ),
        _show_indicator=False,
    )
    assert "Timeout" in model.prompts[0]


def test_label_over_the_cap_explains_itself():
    classifier = _classify(
        {"label": "Refund Delay", "reason": "r", "new_label_description": "d"},
        labels=[Label("pricing", "Asks a price.")],
        allow_none=True,
        label_discovery_config=LabelDiscoveryConfig(
            embedding_model=WordEmbedding(), max_new_labels=0
        ),
    )
    assert classifier.label is None and "max_new_labels" in classifier.reason


def test_merged_duplicate_labels_do_not_repeat_evidence():
    quote = {"field": "input", "quote": "How much is Pro"}
    classifier = _classify(
        {
            "labels": [
                {"label": "pricing", "reason": "r", "evidence": [quote]},
                {"label": "PRICING ", "reason": "r", "evidence": [quote]},
            ]
        },
        multi_label=True,
        evidence_config=EvidenceConfig(),
    )
    assert len(classifier.classifications[0].evidence) == 1
