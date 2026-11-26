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
from deepeval.benchmarks.tasks import BigBenchHardTask
from deepeval.benchmarks.human_eval.human_eval import HumanEval
from deepeval.benchmarks.human_eval.task import HumanEvalTask
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
# HumanEval: generate_samples may return (samples, cost) instead of a list
# --------------------------------------------------------------------------- #

_CORRECT_SAMPLE = "def add(a,b):\n    return a+b"


class _FakeSamplesModel:
    def __init__(self, as_tuple: bool):
        self.as_tuple = as_tuple

    def generate_samples(self, prompt, n, temperature):
        samples = [_CORRECT_SAMPLE] * n
        return (samples, None) if self.as_tuple else samples


def _human_eval() -> HumanEval:
    bench = HumanEval.__new__(HumanEval)
    bench.n = 2
    bench.temperature = 0
    bench.c = {}
    bench.functions = {}
    bench.scorer = Scorer()
    return bench


@pytest.mark.parametrize("as_tuple", [True, False])
def test_humaneval_predict_scores_correct_samples_for_tuple_or_list(as_tuple):
    bench = _human_eval()
    golden = Golden(
        input="add two numbers", expected_output="assert add(1,2)==3"
    )

    result = bench.predict(
        _FakeSamplesModel(as_tuple),
        HumanEvalTask.HAS_CLOSE_ELEMENTS,
        golden,
        k=1,
    )

    assert result["score"] == 1.0
    assert result["prediction"] == [_CORRECT_SAMPLE] * 2
