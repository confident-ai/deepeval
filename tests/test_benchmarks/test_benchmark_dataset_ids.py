"""
Hugging Face dataset ids requested by the benchmarks.

huggingface_hub 1.16 and later only resolve dataset ids of the form
"namespace/name": a legacy id such as "gsm8k" raises HfUriError inside
`load_dataset`, before any data is read.

Offline: `datasets` is replaced by a stub that records the requested ids.
"""

import sys
import types

import pytest

from deepeval.benchmarks import ARC, GSM8K, BoolQ, HumanEval, TruthfulQA
from deepeval.benchmarks.arc.mode import ARCMode
from deepeval.benchmarks.human_eval.task import HumanEvalTask
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.truthful_qa.task import TruthfulQATask


class _StopLoading(Exception):
    pass


class _UnreadDataset:
    """Stands in for the loaded DatasetDict. Selecting a split works, and the
    first read of rows raises _StopLoading, so the test ends right after the
    load_dataset() calls."""

    def __getitem__(self, split):
        return self

    def __iter__(self):
        raise _StopLoading

    def filter(self, *args, **kwargs):
        raise _StopLoading

    def to_pandas(self):
        raise _StopLoading


@pytest.fixture
def requested_ids(monkeypatch):
    ids = []

    def load_dataset(path, *args, **kwargs):
        ids.append(path)
        return _UnreadDataset()

    stub = types.ModuleType("datasets")
    stub.load_dataset = load_dataset
    stub.Dataset = object
    monkeypatch.setitem(sys.modules, "datasets", stub)
    return ids


@pytest.mark.parametrize(
    "load, expected_ids",
    [
        (
            lambda: ARC().load_benchmark_dataset(ARCMode.EASY),
            ["allenai/ai2_arc"],
        ),
        (lambda: GSM8K().load_benchmark_dataset(), ["openai/gsm8k"]),
        (
            lambda: TruthfulQA().load_benchmark_dataset(
                TruthfulQATask.LANGUAGE, TruthfulQAMode.MC1
            ),
            ["truthfulqa/truthful_qa", "truthfulqa/truthful_qa"],
        ),
        (lambda: BoolQ().load_benchmark_dataset(), ["google/boolq"]),
        (
            lambda: HumanEval().load_benchmark_dataset(
                HumanEvalTask.HAS_CLOSE_ELEMENTS
            ),
            ["openai/openai_humaneval"],
        ),
    ],
    ids=["ARC", "GSM8K", "TruthfulQA", "BoolQ", "HumanEval"],
)
def test_benchmark_requests_namespaced_dataset_id(
    requested_ids, load, expected_ids
):
    with pytest.raises(_StopLoading):
        load()
    assert requested_ids == expected_ids
