import json

from deepeval.classifiers import Classifier, EvidenceConfig
from deepeval.classifiers.classifier.steps.truncate import (
    SPAN_FIELD_CHAR_LIMIT,
    SPAN_LIMIT,
    TRUNCATION_MARKER,
    fit_trace,
    fit_turns,
)
from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn
from tests.test_classifiers.fakes import ReplyModel


def _prompt(test_case, **kwargs):
    model = ReplyModel(kwargs.pop("reply", None))
    classifier = Classifier(
        name="topic",
        labels=["pricing"],
        model=model,
        async_mode=False,
        **kwargs,
    )
    classifier.classify(test_case, _show_indicator=False)
    return classifier, model.prompts[0]


def _trace(prompt):
    shown = prompt.split("Trace:\n", 1)[1].split("\n\n---", 1)[0]
    return json.loads(shown)


def _traced(children):
    test_case = LLMTestCase(input="q", actual_output="a")
    test_case._trace_dict = {
        "name": "root",
        "type": "agent",
        "children": children,
    }
    return test_case


def test_long_span_fields_are_capped_with_the_marker():
    _, prompt = _prompt(
        _traced(
            [
                {
                    "name": "llm",
                    "type": "llm",
                    "output": "x" * 5000,
                    "children": [],
                }
            ]
        )
    )
    output = _trace(prompt)["spans"][1]["output"]
    assert output == "x" * SPAN_FIELD_CHAR_LIMIT + TRUNCATION_MARKER


def test_only_the_highest_priority_spans_are_kept_in_order():
    children = [
        {"name": f"base{i}", "type": "base", "children": []}
        for i in range(SPAN_LIMIT)
    ] + [{"name": "failed", "type": "tool", "error": "boom", "children": []}]
    spans = _trace(_prompt(_traced(children))[1])["spans"]
    assert len(spans) == SPAN_LIMIT
    assert spans[0]["name"] == "root" and spans[-1]["name"] == "failed"
    assert [s["index"] for s in spans] == [
        *range(1, SPAN_LIMIT),
        SPAN_LIMIT + 2,
    ]


def test_trace_over_budget_drops_low_priority_spans_first():
    trace = {
        "input": "q",
        "spans": [
            {"index": 1, "name": "a", "type": "base", "output": "y" * 200},
            {"index": 2, "name": "b", "type": "llm", "output": "z" * 50},
        ],
    }
    fitted = fit_trace(trace, 160)
    assert [s["name"] for s in fitted["spans"]] == ["b"]
    assert fit_trace({"input": "q" * 40}, 30) == (
        '{"input":"qqqqqqqq' + TRUNCATION_MARKER
    )


def test_conversation_over_budget_keeps_latest_turns():
    turns = [{"role": "user", "content": str(i) * 50} for i in range(5)]
    kept = fit_turns(turns, len(str(turns[-2:])))
    assert kept == turns[-2:]
    assert fit_turns(turns[:1], 1)[0]["content"] == TRUNCATION_MARKER


def test_single_turn_content_is_cut_to_budget():
    _, prompt = _prompt(LLMTestCase(input="q" * 500), max_content_chars=50)
    shown = prompt.split("Test Case:\n", 1)[1].split("\n\n---", 1)[0]
    assert len(shown) == 50 and shown.endswith(TRUNCATION_MARKER)


def test_evidence_turn_index_points_at_the_original_turn():
    turns = [
        Turn(role="user", content=f"turn number {i} " * 20) for i in range(5)
    ]
    classifier, _ = _prompt(
        ConversationalTestCase(turns=turns),
        max_content_chars=len(
            str([{"role": "user", "content": turns[-1].content}])
        )
        + 5,
        evidence_config=EvidenceConfig(),
        reply={
            "label": "pricing",
            "reason": "r",
            "evidence": [
                {"field": "content", "quote": "turn number 4", "turn_index": 1}
            ],
        },
    )
    [evidence] = classifier.classifications[0].evidence
    assert evidence.matched and evidence.turn_index == 5
