import pytest

from deepeval.classifiers import (
    Classifier,
    EvidenceConfig,
    Label,
    LabelDiscoveryConfig,
)
from deepeval.test_case import LLMTestCase
from tests.test_classifiers.fakes import ReplyModel

TEST_CASE = LLMTestCase(
    input="How much is Pro, and can you cancel my order?",
    actual_output="Pro is $99. I can't help with orders.",
)
LABELS = [
    Label("pricing", "Asks a price."),
    Label("cancellation", "Asks to cancel."),
]


def _classify(reply, **kwargs):
    model = ReplyModel(reply)
    classifier = Classifier(
        name="topic",
        labels=LABELS,
        model=model,
        async_mode=False,
        multi_label=True,
        **kwargs
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
    assert [r.label for r in classifier.classifications] == [
        "pricing",
        "cancellation",
    ]
    assert (
        classifier.label == "pricing"
        and classifier.reason == "Asked the Pro price."
    )
    assert '"labels"' in prompt and "every label that applies" in prompt
    assert classifier.is_successful("cancellation")


def test_repeated_labels_merge_their_evidence():
    classifier, _ = _classify(
        {
            "labels": [
                {
                    "label": "pricing",
                    "reason": "a",
                    "evidence": [
                        {"field": "input", "quote": "How much is Pro"}
                    ],
                },
                {
                    "label": "pricing",
                    "reason": "b",
                    "evidence": [
                        {"field": "actual_output", "quote": "Pro is $99."}
                    ],
                },
            ]
        },
        evidence_config=EvidenceConfig(),
    )
    [result] = classifier.classifications
    assert [e.matched for e in result.evidence] == [True, True]


def test_none_item_gives_no_labels_and_its_reason():
    classifier, _ = _classify(
        {"labels": [{"label": "NONE", "reason": "Just a greeting."}]},
        allow_none=True,
    )
    assert classifier.classifications == [] and classifier.label is None
    assert classifier.reason == "Just a greeting." and classifier.error is None


def test_only_unknown_labels_is_an_error():
    classifier, _ = _classify({"labels": [{"label": "billing", "reason": "x"}]})
    assert classifier.classifications == [] and classifier.error is not None


def test_single_object_reply_is_accepted():
    classifier, _ = _classify({"label": "pricing", "reason": "Asked a price."})
    assert [r.label for r in classifier.classifications] == ["pricing"]


def test_multi_label_with_new_labels():
    classifier, _ = _classify(
        {
            "labels": [
                {
                    "label": "pricing",
                    "reason": "Price.",
                    "new_label_description": None,
                },
                {
                    "label": "Order Help",
                    "reason": "Order.",
                    "new_label_description": "Cases about orders.",
                },
            ]
        },
        label_discovery_config=LabelDiscoveryConfig(),
    )
    assert [
        (r.label, r.new_label_description) for r in classifier.classifications
    ] == [
        ("pricing", None),
        ("Order Help", "Cases about orders."),
    ]


def test_single_label_mode_fills_results():
    model = ReplyModel({"label": "pricing", "reason": "Price."})
    classifier = Classifier(
        name="topic", labels=LABELS, model=model, async_mode=False
    )
    classifier.classify(TEST_CASE, _show_indicator=False)
    assert [r.label for r in classifier.classifications] == ["pricing"]
    assert 'labels"' not in model.prompts[0]


def test_evaluate_counts_one_verdict_per_test_case():
    from deepeval.evaluate.configs import DisplayConfig
    from deepeval.evaluate.console_report import _aggregate_classifications
    from deepeval.evaluate.evaluate import evaluate

    classifier = Classifier(
        name="topic",
        labels=["pricing", "refund", "other"],
        model=ReplyModel(
            {
                "labels": [
                    {"label": "pricing", "reason": "r"},
                    {"label": "refund", "reason": "r"},
                ]
            }
        ),
        multi_label=True,
        async_mode=False,
    )
    result = evaluate(
        test_cases=[
            LLMTestCase(
                input="q",
                actual_output="a",
                expected_labels={"topic": "refund"},
            ),
            LLMTestCase(
                input="q", actual_output="a", expected_labels={"topic": "other"}
            ),
        ],
        classifiers=[classifier],
        display_config=DisplayConfig(print_results=False, show_indicator=False),
    )
    assert [r.success for r in result.test_results] == [True, False]
    aggregate = _aggregate_classifications(result.test_results)["topic"]
    assert (aggregate["total"], aggregate["passes"], aggregate["fails"]) == (
        2,
        1,
        1,
    )
    assert aggregate["labels"] == {"pricing": 2, "refund": 2}


@pytest.mark.parametrize(
    "expected, success",
    [
        ("refund", True),
        ("other", False),
        (["pricing", "refund"], True),
        (["refund"], False),
        (["pricing", "refund", "other"], False),
    ],
)
def test_expected_label_or_exact_list(expected, success):
    classifier = Classifier(
        name="topic",
        labels=["pricing", "refund", "other"],
        model=ReplyModel(
            {
                "labels": [
                    {"label": "pricing", "reason": "r"},
                    {"label": "refund", "reason": "r"},
                ]
            }
        ),
        multi_label=True,
        async_mode=False,
    )
    classifier.classify(TEST_CASE, _show_indicator=False)
    assert classifier.is_successful(expected) is success


def test_a_list_is_rejected_for_a_single_label_classifier():
    from deepeval.classifiers.utils import validate_classifiers

    classifier = Classifier(
        name="topic", labels=["pricing", "refund"], model=ReplyModel({})
    )
    with pytest.raises(ValueError, match="multi_label"):
        validate_classifiers(
            [
                LLMTestCase(
                    input="q",
                    actual_output="a",
                    expected_labels={"topic": ["pricing"]},
                )
            ],
            [classifier],
        )
