"""
Regression tests for silent answer-scoring bugs in the benchmarks.

Each test targets a case where a *correct* prediction was previously scored 0
without any error being raised. They are offline: no model, network, dataset
download, or API key required.
"""

import pytest

from deepeval.scorer.scorer import Scorer
from deepeval.benchmarks.schema import (
    ARCMultipleChoiceSchema,
    MultipleChoiceSchema,
    MultipleChoiceSchemaLower,
)
from deepeval.benchmarks.arc.template import ARCTemplate
from deepeval.benchmarks.drop.template import DROPTemplate
from deepeval.benchmarks.drop.drop import DELIMITER
from deepeval.benchmarks.big_bench_hard.big_bench_hard import BigBenchHard
from deepeval.benchmarks.tasks import BigBenchHardTask
from deepeval.dataset import Golden

# --------------------------------------------------------------------------- #
# MathQA: the answer schema must be able to represent option "e"
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("option", ["a", "b", "c", "d", "e"])
def test_mathqa_schema_accepts_all_five_options(option):
    # MathQA (AQuA-RAT) questions always have options a-e. Previously the schema
    # was Literal["a","b","c","d"], so a schema-constrained model could never
    # emit "e" and every "e"-answer item was scored 0.
    assert MultipleChoiceSchemaLower(answer=option).answer == option


# --------------------------------------------------------------------------- #
# DROP: packing/unpacking answer spans must not corrupt answers with commas
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "span",
    ["1,000", "25,000", "1,234,567", "New York, New York"],
)
def test_drop_single_span_with_comma_survives_round_trip(span):
    packed = DROPTemplate.parse_list_to_str([span], DELIMITER)
    unpacked = DROPTemplate.parse_str_to_list(packed, DELIMITER)
    assert unpacked == [span]
    # a perfect prediction of the gold span must score 1
    assert Scorer.quasi_contains_score(unpacked, span) == 1


def test_drop_multi_span_still_splits():
    spans = ["Paris", "London"]
    packed = DROPTemplate.parse_list_to_str(spans, DELIMITER)
    unpacked = DROPTemplate.parse_str_to_list(packed, DELIMITER)
    assert unpacked == spans
    assert Scorer.quasi_contains_score(unpacked, "London") == 1


def test_drop_delimiter_is_not_a_character_that_occurs_in_answers():
    # The separator must not be something that can appear inside a DROP answer.
    assert DELIMITER not in "0123456789,.- abcdefghijklmnopqrstuvwxyz"


# --------------------------------------------------------------------------- #
# BigBenchHard: the batch path must not corrupt schema-constrained answers
# --------------------------------------------------------------------------- #


class _FakeSchemaBatchModel:
    """A native/schema model: batch_generate(schemas=...) returns schema
    instances. Simulates a perfect model that always selects "(A)"."""

    def get_model_name(self):
        return "fake"

    def batch_generate(self, prompts, schemas=None):
        if schemas is None:
            # signal "no schema support" so callers fall back to free text
            raise TypeError("schema-less generation not supported")
        return [schema(answer="(A)") for schema in schemas]


def _bbh(enable_cot: bool) -> BigBenchHard:
    # Bypass __init__ (which imports the optional HF `datasets` package); the
    # batch path only needs these three attributes.
    bench = BigBenchHard.__new__(BigBenchHard)
    bench.n_shots = 0
    bench.enable_cot = enable_cot
    bench.scorer = Scorer()
    return bench


@pytest.mark.parametrize("enable_cot", [True, False])
def test_bbh_batch_predict_scores_schema_answer_correctly(enable_cot):
    bench = _bbh(enable_cot=enable_cot)
    goldens = [Golden(input="pick the right sentence", expected_output="(A)")]

    result = bench.batch_predict(
        _FakeSchemaBatchModel(), BigBenchHardTask.HYPERBATON, goldens
    )

    # Previously, with enable_cot=True the batch path ran prediction[:-1] on the
    # schema answer, turning "(A)" into "(A)" -> "(A" and scoring it 0.
    assert result[0]["prediction"] == "(A)"
    assert result[0]["score"] == 1


# --------------------------------------------------------------------------- #
# ARC: the gold answer must be expressible in the alphabet the model is given
# --------------------------------------------------------------------------- #

# A numerically labelled item, as it appears in ai2_arc. 127 of the items in the
# train splits (23 in ARC-Challenge, 104 in ARC-Easy) label their options this
# way or offer five of them.
ARC_NUMERIC_ITEM = {
    "question": "Rocks are classified as igneous, metamorphic, or sedimentary according to",
    "choices": {
        "text": [
            "their color",
            "their shape",
            "how they formed",
            "the minerals they contain",
        ],
        "label": ["1", "2", "3", "4"],
    },
    "answerKey": "3",
}

ARC_LETTER_ITEM = {
    "question": "Which factor will most likely cause a person to develop a fever?",
    "choices": {
        "text": [
            "a leg muscle relaxing after exercise",
            "a bacterial population in the bloodstream",
            "several viral particles on the skin",
            "carbohydrates being digested in the stomach",
        ],
        "label": ["A", "B", "C", "D"],
    },
    "answerKey": "B",
}

ARC_FIVE_OPTION_ITEM = {
    "question": "Which is a renewable resource?",
    "choices": {
        "text": ["coal", "oil", "natural gas", "uranium", "timber"],
        "label": ["A", "B", "C", "D", "E"],
    },
    "answerKey": "E",
}

ARC_THREE_OPTION_ITEM = {
    "question": "Which state of matter has a fixed volume but no fixed shape?",
    "choices": {
        "text": ["solid", "liquid", "gas"],
        "label": ["1", "2", "3"],
    },
    "answerKey": "2",
}


@pytest.mark.parametrize("option", ["A", "B", "C", "D", "E"])
def test_arc_schema_accepts_all_five_options(option):
    # A subset of ARC items have five options, so "E" is a legitimate gold
    # answer. ARC keeps its own schema rather than widening the shared
    # MultipleChoiceSchema, which MMLU/HellaSwag/LogiQA rely on being A-D.
    assert ARCMultipleChoiceSchema(answer=option).answer == option


def test_shared_multiple_choice_schema_stays_four_options():
    # Guard against "fixing" ARC by loosening the schema the strictly
    # four-option benchmarks depend on.
    with pytest.raises(Exception):
        MultipleChoiceSchema(answer="E")


@pytest.mark.parametrize(
    "data, expected",
    [
        (ARC_NUMERIC_ITEM, "C"),
        (ARC_LETTER_ITEM, "B"),
        (ARC_FIVE_OPTION_ITEM, "E"),
        (ARC_THREE_OPTION_ITEM, "B"),
    ],
)
def test_arc_gold_answer_is_normalized_to_its_positional_letter(data, expected):
    # Previously format_answer returned data["answerKey"] verbatim, so a
    # numerically labelled item had expected_output "3" while the model was
    # constrained to "A"-"D" — unmatchable, and scored 0 in silence.
    assert ARCTemplate.format_answer(data) == expected


@pytest.mark.parametrize(
    "data",
    [
        ARC_NUMERIC_ITEM,
        ARC_LETTER_ITEM,
        ARC_FIVE_OPTION_ITEM,
        ARC_THREE_OPTION_ITEM,
    ],
)
def test_arc_gold_answer_is_always_schema_representable(data):
    # The invariant the bug violated: whatever ARC stores as the expected output
    # must be something a schema-constrained model is able to emit.
    gold = ARCTemplate.format_answer(data)
    assert ARCMultipleChoiceSchema(answer=gold).answer == gold


@pytest.mark.parametrize(
    "data",
    [
        ARC_NUMERIC_ITEM,
        ARC_LETTER_ITEM,
        ARC_FIVE_OPTION_ITEM,
        ARC_THREE_OPTION_ITEM,
    ],
)
def test_arc_prompt_offers_the_gold_answer_as_a_choice(data):
    # The rendered options and the gold answer must use the same alphabet,
    # otherwise the prompt asks for one thing and scoring expects another.
    prompt = ARCTemplate.format_question(data, include_answer=False)
    gold = ARCTemplate.format_answer(data)
    offered = [
        line.split(".", 1)[0]
        for line in prompt.splitlines()
        if len(line) > 1 and line[1] == "."
    ]
    assert offered == ARCTemplate.option_letters[: len(data["choices"]["text"])]
    assert gold in offered


def test_arc_letter_labelled_items_are_unchanged():
    # The common case must render exactly as before this fix.
    assert ARCTemplate.format_question(ARC_LETTER_ITEM, False) == (
        "Which factor will most likely cause a person to develop a fever?"
        "\nA. a leg muscle relaxing after exercise"
        "\nB. a bacterial population in the bloodstream"
        "\nC. several viral particles on the skin"
        "\nD. carbohydrates being digested in the stomach"
        "\nAnswer: "
    )


def test_arc_few_shot_prompt_does_not_teach_an_unrepresentable_answer():
    # The fifth built-in n-shot example is numerically labelled, so with the
    # default n_shots=5 every ARC prompt demonstrated "Answer:  3" — an answer
    # the schema forbids the model from giving.
    prompt = ARCTemplate.generate_output(input="", n_shots=5)
    for shot in ARCTemplate.n_shot_examples:
        demonstrated = ARCTemplate.format_answer(shot)
        assert (
            ARCMultipleChoiceSchema(answer=demonstrated).answer == demonstrated
        )
    assert "Answer:  3\n" not in prompt


def test_arc_correct_prediction_on_a_numeric_item_now_scores_one():
    # End to end: the model picks the right option, names it in the alphabet the
    # prompt gave it, and is scored correctly. Before the fix the gold was "3",
    # the model could only say "C", and exact_match_score returned 0.
    gold = ARCTemplate.format_answer(ARC_NUMERIC_ITEM)
    prediction = ARCMultipleChoiceSchema(answer="C").answer
    assert Scorer.exact_match_score(gold, prediction) == 1


def test_arc_normalize_label_passes_through_unknown_labels():
    # An unexpected dataset shape degrades to the old behaviour, never raises.
    assert ARCTemplate.normalize_label(ARC_LETTER_ITEM, "Z") == "Z"
