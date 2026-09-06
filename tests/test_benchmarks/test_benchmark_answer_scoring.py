"""
Regression tests for silent answer-scoring bugs in the benchmarks.

Each test targets a case where a *correct* prediction was previously scored 0
without any error being raised. They are offline: no model, network, dataset
download, or API key required.
"""

import pytest

from deepeval.scorer.scorer import Scorer
from deepeval.benchmarks.schema import MultipleChoiceSchemaLower
from deepeval.benchmarks.drop.template import DROPTemplate
from deepeval.benchmarks.drop.drop import DELIMITER
from deepeval.benchmarks.big_bench_hard.big_bench_hard import BigBenchHard
from deepeval.benchmarks.truthful_qa.truthful_qa import (
    TruthfulQA,
    truthful_qa_confinement_statements_dict,
)
from deepeval.benchmarks.modes import TruthfulQAMode
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
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
# TruthfulQA MC2: the per-question score must be an accuracy fraction, because
# `evaluate` sums it and divides by the question count
# --------------------------------------------------------------------------- #


class _FakeMC2Model:
    """A native/schema model that always selects `indices` for MC2."""

    def __init__(self, indices):
        self.indices = indices

    def get_model_name(self):
        return "fake"

    def generate(self, prompt, schema=None):
        if schema is None:
            raise TypeError("schema-less generation not supported")
        return schema(answer=self.indices)

    def batch_generate(self, prompts, schemas=None):
        if schemas is None:
            raise TypeError("schema-less generation not supported")
        return [schema(answer=self.indices) for schema in schemas]


def _tqa() -> TruthfulQA:
    # Bypass __init__, which imports the optional HF `datasets` package.
    bench = TruthfulQA.__new__(TruthfulQA)
    bench.scorer = Scorer()
    bench.confinement_instructions_dict = (
        truthful_qa_confinement_statements_dict
    )
    return bench


@pytest.mark.parametrize(
    "expected,predicted,score",
    [
        ("[1, 2]", [1, 2], 1.0),
        ("[1, 2]", [1], 0.5),
        ("[1, 2]", [3], 0.0),
        ("[1, 2, 3, 4]", [1, 2, 3], 0.75),
    ],
)
def test_truthfulqa_mc2_score_is_a_fraction(expected, predicted, score):
    # `evaluate` computes accuracy as sum(score) / len(goldens), so a question
    # must contribute at most 1. truth_identification_score returns a 0-100
    # percentage, which made a perfect MC2 run report an accuracy of 100.0.
    golden = Golden(input="q", expected_output=expected)
    result = _tqa().predict(
        _FakeMC2Model(predicted), golden, TruthfulQAMode.MC2
    )
    assert result["score"] == pytest.approx(score)


def test_truthfulqa_mc2_batch_and_single_scores_agree():
    golden = Golden(input="q", expected_output="[1, 2]")
    model = _FakeMC2Model([1])

    single = _tqa().predict(model, golden, TruthfulQAMode.MC2)["score"]
    batched = _tqa().batch_predict(model, [golden], TruthfulQAMode.MC2)[0][
        "score"
    ]

    assert single == pytest.approx(batched)


def test_truthfulqa_mc1_score_still_binary():
    golden = Golden(input="q", expected_output="2")

    class _FakeMC1Model(_FakeMC2Model):
        def generate(self, prompt, schema=None):
            return schema(answer=self.indices)

    assert (
        _tqa().predict(_FakeMC1Model(2), golden, TruthfulQAMode.MC1)["score"]
        == 1
    )


def _tqa_for_evaluate(goldens) -> TruthfulQA:
    bench = _tqa()
    bench.tasks = [TruthfulQATask.LANGUAGE]
    bench.mode = TruthfulQAMode.MC2
    bench.n_problems_per_task = None
    bench.verbose_mode = False
    bench.mc_dataset = None
    bench.load_benchmark_dataset = lambda task, mode: goldens
    return bench


@pytest.mark.parametrize("batch_size", [None, 2])
def test_truthfulqa_mc2_evaluate_accuracy_is_an_accuracy(
    monkeypatch, batch_size
):
    # A model that identifies 1 of the 2 correct answers on every question
    # should score 0.5, whether or not the batch path is used. Previously the
    # non-batch path summed percentages (0.5 -> 50.0) and the batch path gave
    # full credit for any non-zero score (0.5 -> 1.0).
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    goldens = [Golden(input="q", expected_output="[1, 2]") for _ in range(4)]

    result = _tqa_for_evaluate(goldens).evaluate(
        _FakeMC2Model([1]), batch_size=batch_size
    )

    assert result.overall_accuracy == pytest.approx(0.5)


def test_truthfulqa_mc2_repeated_indices_count_once():
    # Nothing stops a model repeating an index: the schema is a plain
    # List[int] and the free-text fallback parses whatever it is given.
    # Counting each repeat as a match gave [1, 1, 1] against [1, 2] full
    # credit for finding one of the two correct answers.
    golden = Golden(input="q", expected_output="[1, 2]")
    result = _tqa().predict(
        _FakeMC2Model([1, 1, 1]), golden, TruthfulQAMode.MC2
    )
    assert result["score"] == pytest.approx(0.5)
