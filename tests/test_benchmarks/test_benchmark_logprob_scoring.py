"""
Offline tests for optional log-probability scoring in the benchmarks
(deepeval issue #2311). No model, network, dataset download, or API key
required.
"""

import pytest

from deepeval.scorer.scorer import Scorer
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.dataset import Golden
from deepeval.errors import DeepEvalError


# --------------------------------------------------------------------------- #
# Fakes: mimic the shape of an OpenAI ChatCompletion carrying top_logprobs
# --------------------------------------------------------------------------- #


class _TopLogprob:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


class _Content:
    def __init__(self, top_logprobs):
        self.top_logprobs = top_logprobs


class _Logprobs:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, logprobs):
        self.logprobs = logprobs


class _Completion:
    def __init__(self, first_token_top_logprobs):
        self.choices = [
            _Choice(_Logprobs([_Content(first_token_top_logprobs)]))
        ]


class _FakeLogprobModel:
    def __init__(self, top_logprobs, supports=True):
        self._top_logprobs = top_logprobs
        self._supports = supports

    def get_model_name(self):
        return "fake-logprob-model"

    def supports_log_probs(self):
        return self._supports

    def generate_raw_response(self, prompt, top_logprobs=5):
        return _Completion(self._top_logprobs), 0.0


def _mmlu(scoring_mode="logprobs"):
    # Bypass __init__ (it imports the optional `datasets`/`pandas` deps); set
    # only the attributes the log-prob predict path touches.
    bench = MMLU.__new__(MMLU)
    bench.scorer = Scorer()
    bench.scoring_mode = scoring_mode
    bench.top_logprobs = 20
    bench.choices = ["A", "B", "C", "D"]
    bench.n_shots = 0  # template never indexes the shots set
    bench.shots_dataset = [{}]  # non-empty (passes the assert)
    bench.confinement_instructions = "Output 'A', 'B', 'C', or 'D'."
    return bench


# --------------------------------------------------------------------------- #
# Scorer.best_choice_from_logprobs
# --------------------------------------------------------------------------- #


def test_best_choice_picks_highest_logprob():
    top = [
        _TopLogprob("A", -2.1),
        _TopLogprob("B", -0.05),
        _TopLogprob("C", -3.0),
    ]
    assert Scorer.best_choice_from_logprobs(top, ["A", "B", "C", "D"]) == "B"


def test_best_choice_matches_token_ignoring_whitespace_and_case():
    # Providers often emit the option with a leading space / different case.
    top = [{"token": " b", "logprob": -0.1}, {"token": "A", "logprob": -0.9}]
    assert Scorer.best_choice_from_logprobs(top, ["A", "B", "C", "D"]) == "B"


def test_best_choice_accepts_tuple_pairs():
    top = [("A", -0.3), ("B", -1.2)]
    assert Scorer.best_choice_from_logprobs(top, ["A", "B", "C", "D"]) == "A"


def test_best_choice_returns_none_when_no_candidate_present():
    top = [_TopLogprob("hello", -0.1), _TopLogprob("world", -0.2)]
    assert Scorer.best_choice_from_logprobs(top, ["A", "B", "C", "D"]) is None


# --------------------------------------------------------------------------- #
# MMLU log-prob predict path
# --------------------------------------------------------------------------- #


def test_mmlu_logprob_predict_scores_correct_answer():
    # Most probability mass on "C"; golden is "C" -> score 1.
    model = _FakeLogprobModel(
        [
            _TopLogprob("A", -3.0),
            _TopLogprob("C", -0.02),
            _TopLogprob("B", -2.0),
        ]
    )
    bench = _mmlu()
    golden = Golden(
        input="Q\nA. w\nB. x\nC. y\nD. z\nAnswer:", expected_output="C"
    )
    out = bench.predict(model, list(MMLUTask)[0], golden)
    assert out["prediction"] == "C"
    assert out["score"] == 1


def test_mmlu_logprob_predict_marks_wrong_answer():
    model = _FakeLogprobModel([_TopLogprob("A", -0.02), _TopLogprob("C", -3.0)])
    bench = _mmlu()
    golden = Golden(input="Q\nAnswer:", expected_output="C")
    out = bench.predict(model, list(MMLUTask)[0], golden)
    assert out["prediction"] == "A"
    assert out["score"] == 0


def test_mmlu_logprob_guard_raises_for_unsupported_model():
    model = _FakeLogprobModel([], supports=False)
    bench = _mmlu()
    golden = Golden(input="Q\nAnswer:", expected_output="A")
    with pytest.raises(DeepEvalError):
        bench.predict(model, list(MMLUTask)[0], golden)
