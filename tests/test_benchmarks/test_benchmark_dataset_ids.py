"""
Regression tests for the Hugging Face dataset ids used by the benchmarks.

huggingface_hub 1.16 removed support for legacy single-segment dataset ids
(e.g. "gsm8k"), so every benchmark dataset id must be namespaced
(e.g. "openai/gsm8k"); otherwise load_dataset raises HfUriError before any
data is downloaded. They are offline: no model, network, dataset download,
or API key required - datasets.load_dataset is stubbed and the captured
ids and configs are asserted.
"""

import sys
import types

import pandas as pd
import pytest

ARC_ROW = {
    "id": "test",
    "question": "Which is correct?",
    "choices": {"text": ["option a", "option b"], "label": ["A", "B"]},
    "answerKey": "A",
}
GSM8K_ROW = {"question": "What is 1 + 1?", "answer": "Adding gives 2.\n#### 2"}
TRUTHFUL_QA_ROW = {
    "question": "What is 1 + 1?",
    "category": "Science",
    "best_answer": "2",
    "correct_answers": ["2"],
    "incorrect_answers": ["3"],
    "mc1_targets": {"choices": ["2", "3"], "labels": [1, 0]},
    "mc2_targets": {"choices": ["2", "3"], "labels": [1, 0]},
}
BOOLQ_ROW = {
    "question": "is the sky blue",
    "passage": "The sky is blue.",
    "answer": True,
}
HUMAN_EVAL_ROW = {
    "task_id": "Test/0",
    "entry_point": "has_close_elements",
    "prompt": "def has_close_elements():\n",
    "test": "def check(candidate):\n",
    "canonical_solution": "    pass\n",
}

ROW_BY_DATASET_ID = {
    "allenai/ai2_arc": ARC_ROW,
    "openai/gsm8k": GSM8K_ROW,
    "truthfulqa/truthful_qa": TRUTHFUL_QA_ROW,
    "google/boolq": BOOLQ_ROW,
    "openai/openai_humaneval": HUMAN_EVAL_ROW,
}


class FakeSplit(list):
    """Minimal stand-in for datasets.Dataset split objects."""

    def __init__(self, row):
        super().__init__([row])
        self._row = row

    def to_pandas(self):
        return pd.DataFrame([self._row])

    def filter(self, fn):
        return self


class FakeDataset:
    """Minimal stand-in for datasets.DatasetDict / datasets.Dataset."""

    def __init__(self, row):
        self._row = row

    def __getitem__(self, split):
        return FakeSplit(self._row)

    def filter(self, fn):
        return FakeSplit(self._row)

    @classmethod
    def from_pandas(cls, df):
        records = df.to_dict("records")
        return cls(records[0] if records else TRUTHFUL_QA_ROW)


@pytest.fixture
def captured_dataset_ids(monkeypatch):
    calls = []

    def fake_load_dataset(repo_id, config=None, *args, **kwargs):
        calls.append((repo_id, config))
        return FakeDataset(ROW_BY_DATASET_ID[repo_id])

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    fake_datasets.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    return calls


def test_arc_uses_namespaced_dataset_id(captured_dataset_ids):
    from deepeval.benchmarks.arc.arc import ARC
    from deepeval.benchmarks.arc.mode import ARCMode

    goldens = ARC().load_benchmark_dataset(ARCMode.EASY)
    assert len(goldens) == 1
    assert ("allenai/ai2_arc", "ARC-Easy") in captured_dataset_ids

    goldens = ARC(mode=ARCMode.CHALLENGE).load_benchmark_dataset(
        ARCMode.CHALLENGE
    )
    assert len(goldens) == 1
    assert ("allenai/ai2_arc", "ARC-Challenge") in captured_dataset_ids


def test_gsm8k_uses_namespaced_dataset_id(captured_dataset_ids):
    from deepeval.benchmarks.gsm8k.gsm8k import GSM8K

    goldens = GSM8K().load_benchmark_dataset()
    assert len(goldens) == 1
    assert ("openai/gsm8k", "main") in captured_dataset_ids


def test_truthful_qa_uses_namespaced_dataset_id(captured_dataset_ids):
    from deepeval.benchmarks.truthful_qa.truthful_qa import TruthfulQA
    from deepeval.benchmarks.truthful_qa.task import TruthfulQATask
    from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode

    goldens = TruthfulQA(tasks=[TruthfulQATask.SCIENCE]).load_benchmark_dataset(
        TruthfulQATask.SCIENCE, TruthfulQAMode.MC1
    )
    assert len(goldens) == 1
    assert ("truthfulqa/truthful_qa", "generation") in captured_dataset_ids
    assert (
        "truthfulqa/truthful_qa",
        "multiple_choice",
    ) in captured_dataset_ids


def test_bool_q_uses_namespaced_dataset_id(captured_dataset_ids):
    from deepeval.benchmarks.bool_q.bool_q import BoolQ

    goldens = BoolQ().load_benchmark_dataset()
    assert len(goldens) == 1
    assert ("google/boolq", "default") in captured_dataset_ids


def test_human_eval_uses_namespaced_dataset_id(captured_dataset_ids):
    from deepeval.benchmarks.human_eval.human_eval import HumanEval
    from deepeval.benchmarks.human_eval.task import HumanEvalTask

    golden = HumanEval().load_benchmark_dataset(
        HumanEvalTask.HAS_CLOSE_ELEMENTS
    )
    assert golden.expected_output == HUMAN_EVAL_ROW["test"]
    assert ("openai/openai_humaneval", None) in captured_dataset_ids
