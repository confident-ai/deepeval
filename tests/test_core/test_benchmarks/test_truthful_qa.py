"""Parity tests for TruthfulQA MC2 accuracy.

The bug under test: `TruthfulQA.evaluate(model)` and
`TruthfulQA.evaluate(model, batch_size=N)` historically reported wildly
different `overall_accuracy` for the same model and data, because:

- The non-batch path accumulated a 0..100 percentage into `task_correct_predictions`.
- The batch path collapsed any nonzero score to `+1`.

The fix normalizes the scorer return to a fraction in [0, 1] and uses a single
`task_correct_predictions += score` in both branches. These tests pin both the
parity invariant and the hand-computed expected mean so neither path can
silently drift.
"""

from typing import List

import pytest

from deepeval.dataset import Golden
from deepeval.benchmarks.schema import ListOfNumbersSchema
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
from deepeval.benchmarks.truthful_qa.truthful_qa import TruthfulQA
from deepeval.models.base_model import DeepEvalBaseLLM


class _StubLLM(DeepEvalBaseLLM):
    """Deterministic stub returning a fixed `ListOfNumbersSchema` per input.

    Defines both `generate` and `batch_generate` so `should_use_batch` (which
    only checks `hasattr(model, "batch_generate")`) treats it as batch-capable.
    """

    def __init__(self, answers_by_input: dict):
        self._answers = answers_by_input
        self.calls = 0
        self.batch_calls = 0
        # Hardens the stub against future DeepEvalBaseLLM.__init__ changes
        # (e.g. tracing/telemetry hooks that read self.name or self.model).
        super().__init__(model="stub-truthful-qa")

    def load_model(self):
        return self

    def _lookup(self, prompt: str) -> List[int]:
        for key, value in self._answers.items():
            if key in prompt:
                return value
        raise KeyError(f"No stubbed answer matches prompt: {prompt!r}")

    def generate(self, prompt, schema=None):
        self.calls += 1
        return ListOfNumbersSchema(answer=self._lookup(prompt))

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema=schema)

    def batch_generate(self, prompts, schemas=None):
        self.batch_calls += 1
        return [
            ListOfNumbersSchema(answer=self._lookup(p)) for p in prompts
        ]

    def get_model_name(self) -> str:
        return "stub"


def _make_benchmark(goldens: List[Golden]) -> TruthfulQA:
    benchmark = TruthfulQA(
        tasks=[TruthfulQATask.LANGUAGE],
        mode=TruthfulQAMode.MC2,
    )
    # Bypass HuggingFace dataset loading; return fixed goldens for any task.
    benchmark.load_benchmark_dataset = lambda task, mode: list(goldens)
    return benchmark


# Four questions chosen to span the metric's edge cases:
#   - full match (1.0)
#   - half recall (0.5)
#   - zero recall (0.0)
#   - duplicate-in-prediction that pre-fix would inflate to 1.0 but
#     post-fix is the unique-recall fraction 0.25
_GOLDENS = [
    Golden(input="Q: full?", expected_output="[1, 2]"),
    Golden(input="Q: half?", expected_output="[1, 2, 3, 4]"),
    Golden(input="Q: none?", expected_output="[1, 2]"),
    Golden(input="Q: dup?", expected_output="[1, 2, 3, 4]"),
]

_STUB_ANSWERS = {
    "Q: full?": [1, 2],
    "Q: half?": [1, 2],
    "Q: none?": [3, 4],
    "Q: dup?": [1, 1, 1, 1],
}

# Per-question recall:
#   full → 2/2 = 1.0
#   half → 2/4 = 0.5
#   none → 0/2 = 0.0
#   dup  → |{1}∩{1,2,3,4}| / |{1,2,3,4}| = 1/4 = 0.25
_EXPECTED_MEAN = (1.0 + 0.5 + 0.0 + 0.25) / 4  # 0.4375


def test_batch_and_non_batch_overall_accuracy_match():
    """Same model, same data, two code paths — must report identical accuracy."""
    benchmark_seq = _make_benchmark(_GOLDENS)
    benchmark_batch = _make_benchmark(_GOLDENS)

    seq_result = benchmark_seq.evaluate(_StubLLM(_STUB_ANSWERS))
    batch_result = benchmark_batch.evaluate(
        _StubLLM(_STUB_ANSWERS), batch_size=2
    )

    assert seq_result.overall_accuracy == pytest.approx(
        batch_result.overall_accuracy
    )


def test_overall_accuracy_matches_hand_computed_mean():
    """Pin the exact value so both paths can't drift together into a new bug."""
    benchmark_seq = _make_benchmark(_GOLDENS)
    benchmark_batch = _make_benchmark(_GOLDENS)

    seq_result = benchmark_seq.evaluate(_StubLLM(_STUB_ANSWERS))
    batch_result = benchmark_batch.evaluate(
        _StubLLM(_STUB_ANSWERS), batch_size=2
    )

    assert seq_result.overall_accuracy == pytest.approx(_EXPECTED_MEAN)
    assert batch_result.overall_accuracy == pytest.approx(_EXPECTED_MEAN)


def test_overall_accuracy_in_unit_interval():
    """The published contract is `overall_score ∈ [0, 1]`."""
    benchmark = _make_benchmark(_GOLDENS)
    result = benchmark.evaluate(_StubLLM(_STUB_ANSWERS))
    assert 0.0 <= result.overall_accuracy <= 1.0


def test_per_task_scores_match_between_paths():
    benchmark_seq = _make_benchmark(_GOLDENS)
    benchmark_batch = _make_benchmark(_GOLDENS)

    benchmark_seq.evaluate(_StubLLM(_STUB_ANSWERS))
    benchmark_batch.evaluate(_StubLLM(_STUB_ANSWERS), batch_size=2)

    seq_scores = dict(
        zip(benchmark_seq.task_scores["Task"], benchmark_seq.task_scores["Score"])
    )
    batch_scores = dict(
        zip(
            benchmark_batch.task_scores["Task"],
            benchmark_batch.task_scores["Score"],
        )
    )
    assert seq_scores.keys() == batch_scores.keys()
    for task in seq_scores:
        assert seq_scores[task] == pytest.approx(batch_scores[task])


def test_duplicate_prediction_does_not_inflate_per_question_score():
    """Regression test for the dedup bug in `truth_identification_score`.

    Pre-fix, a prediction of `[1, 1, 1, 1]` against target `[1, 2, 3, 4]` would
    score 4/4 = 1.0 (and after the old `*100, round` 100). Post-fix it is the
    unique-recall fraction 0.25.
    """
    only_dup = [Golden(input="Q: dup?", expected_output="[1, 2, 3, 4]")]
    stub = _StubLLM({"Q: dup?": [1, 1, 1, 1]})
    benchmark = _make_benchmark(only_dup)

    result = benchmark.evaluate(stub)
    assert result.overall_accuracy == pytest.approx(0.25)


def test_batch_path_was_actually_taken():
    """Sanity-check that the batch test exercises the batch_predict branch."""
    benchmark = _make_benchmark(_GOLDENS)
    stub = _StubLLM(_STUB_ANSWERS)
    benchmark.evaluate(stub, batch_size=2)
    assert stub.batch_calls > 0
    assert stub.calls == 0


def test_non_batch_path_was_actually_taken():
    """Sanity-check the parity test isn't accidentally running both via batch."""
    benchmark = _make_benchmark(_GOLDENS)
    stub = _StubLLM(_STUB_ANSWERS)
    benchmark.evaluate(stub)
    assert stub.calls == len(_GOLDENS)
    assert stub.batch_calls == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
