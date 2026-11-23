import sys
from types import SimpleNamespace

import pytest

from deepeval.scorer import Scorer


@pytest.fixture
def bert_score_backend(monkeypatch):
    class Scores:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def numpy(self):
            return self

        def tolist(self):
            return self.values

    class BERTScorer:
        def __init__(self, **kwargs):
            pass

        def score(self, cands, refs):
            # Use exact matches as deterministic scores at the optional backend
            # boundary. BERTScore pairs each candidate with one reference/group.
            values = [
                float(cand == ref if isinstance(ref, str) else cand in ref)
                for cand, ref in zip(cands, refs)
            ]
            return Scores(values), Scores(values), Scores(values)

    monkeypatch.setitem(
        sys.modules, "bert_score", SimpleNamespace(BERTScorer=BERTScorer)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
        ),
    )


@pytest.mark.parametrize(
    "references, predictions, expected",
    [
        ("first", ["first", "second"], [1.0, 0.0]),
        (["first"], ["first", "second"], [1.0, 0.0]),
        (
            ["first", "second"],
            ["second", "third", "first"],
            [1.0, 0.0, 1.0],
        ),
        (
            ["first", "second", "third"],
            ["third", "fourth"],
            [1.0, 0.0],
        ),
        (["first", "second"], "second", [1.0]),
        (["first", "second"], ["second", "first"], [0.0, 0.0]),
        ("first", "first", [1.0]),
    ],
    ids=[
        "shared-string-reference",
        "shared-list-reference",
        "fewer-references-than-candidates",
        "more-references-than-candidates",
        "one-candidate-multiple-references",
        "pairwise-references",
        "single-pair",
    ],
)
def test_bert_score_scores_every_candidate(
    bert_score_backend, references, predictions, expected
):
    result = Scorer.bert_score(references, predictions)

    assert result == {
        "bert-precision": expected,
        "bert-recall": expected,
        "bert-f1": expected,
    }
