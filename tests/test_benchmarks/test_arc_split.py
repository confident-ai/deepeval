"""Offline regression tests for ARC evaluation split selection."""

import sys
from types import ModuleType

import pytest

from deepeval.benchmarks.arc.arc import ARC
from deepeval.benchmarks.arc.mode import ARCMode
from deepeval.benchmarks.arc.template import ARCTemplate


def _row(identifier: str, answer: str) -> dict:
    return {
        "id": identifier,
        "question": f"Question from {identifier}?",
        "choices": {
            "text": [
                "first choice",
                "second choice",
                "third choice",
                "fourth choice",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": answer,
    }


@pytest.mark.parametrize("mode", [ARCMode.EASY, ARCMode.CHALLENGE])
def test_arc_scores_test_split_not_few_shot_training_split(monkeypatch, mode):
    # A few-shot exemplar must never become an evaluated golden.
    exemplar = ARCTemplate.n_shot_examples[0]
    held_out = _row("held-out-test-question", "C")
    calls = []
    dataset = {"train": [exemplar], "test": [held_out]}

    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(name, config):
        calls.append((name, config))
        return dataset

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # The loader does not need model/scorer initialization or a real download.
    benchmark = ARC.__new__(ARC)
    goldens = benchmark.load_benchmark_dataset(mode)

    assert calls == [("ai2_arc", mode.value)]
    assert len(goldens) == 1
    assert goldens[0].input == ARCTemplate.format_question(held_out, False)
    assert goldens[0].expected_output == "C"
    assert ARCTemplate.format_question(exemplar, False) not in [
        golden.input for golden in goldens
    ]

    # Preserve the existing per-mode dataset cache.
    assert benchmark.load_benchmark_dataset(mode) == goldens
    assert calls == [("ai2_arc", mode.value)]
