"""Offline regression tests for ARC evaluation split and accuracy."""

import sys
from contextlib import nullcontext
from types import ModuleType

import pytest

from deepeval.benchmarks.arc.arc import ARC
from deepeval.benchmarks.arc.mode import ARCMode
from deepeval.benchmarks.arc.template import ARCTemplate
from deepeval.dataset import Golden


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
    exemplars = ARCTemplate.n_shot_examples
    exemplar_ids = {example["id"] for example in exemplars}
    assert len(exemplar_ids) == 5
    held_out = _row("held-out-test-question", "C")
    calls = []
    dataset = {"train": exemplars, "test": [held_out]}

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

    # Golden has no ARC ID: map the scored question text back to fixture rows.
    scored_inputs = {golden.input for golden in goldens}
    scored_ids = {
        row["id"]
        for split in ("train", "test")
        for row in dataset[split]
        if ARCTemplate.format_question(row, False) in scored_inputs
    }
    assert scored_ids == {held_out["id"]}
    assert scored_ids.isdisjoint(exemplar_ids)

    # Preserve the existing per-mode dataset cache.
    assert benchmark.load_benchmark_dataset(mode) == goldens
    assert calls == [("ai2_arc", mode.value)]


def test_arc_caches_easy_and_challenge_independently(monkeypatch):
    calls = []
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(name, config):
        calls.append((name, config))
        return {"test": [_row(config, "C")]}

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    benchmark = ARC.__new__(ARC)

    easy = benchmark.load_benchmark_dataset(ARCMode.EASY)
    challenge = benchmark.load_benchmark_dataset(ARCMode.CHALLENGE)
    assert easy[0].input != challenge[0].input
    assert benchmark.load_benchmark_dataset(ARCMode.EASY) == easy
    assert benchmark.load_benchmark_dataset(ARCMode.CHALLENGE) == challenge
    assert calls == [
        ("ai2_arc", ARCMode.EASY.value),
        ("ai2_arc", ARCMode.CHALLENGE.value),
    ]


@pytest.mark.parametrize("mode", [ARCMode.EASY, ARCMode.CHALLENGE])
@pytest.mark.parametrize("available, evaluated", [(0, 0), (2, 2), (7, 5)])
def test_arc_accuracy_uses_actual_evaluated_count(
    monkeypatch, mode, available, evaluated
):
    benchmark = ARC.__new__(ARC)
    benchmark.mode = mode
    benchmark.n_problems = 5
    benchmark.verbose_mode = False
    goldens = [
        Golden(input=f"Question {i}", expected_output="C")
        for i in range(available)
    ]
    benchmark.load_benchmark_dataset = lambda selected_mode: goldens
    benchmark.predict = lambda model, golden: {
        "prediction": golden.expected_output,
        "score": 1,
    }

    telemetry_calls = []

    def fake_capture_benchmark_run(name, count):
        telemetry_calls.append((name, count))
        return nullcontext()

    monkeypatch.setattr(
        "deepeval.benchmarks.arc.arc.capture_benchmark_run",
        fake_capture_benchmark_run,
    )
    result = benchmark.evaluate(model=object())

    expected_accuracy = 1.0 if evaluated else 0.0
    assert result.overall_accuracy == expected_accuracy
    assert benchmark.overall_score == expected_accuracy
    assert len(benchmark.predictions) == evaluated
    assert benchmark.predictions["Correct"].tolist() == [1] * evaluated
    assert telemetry_calls == [("ARC", 5)]
