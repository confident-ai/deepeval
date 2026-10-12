"""
Regression tests for MMLU few-shot leakage across tasks.

MMLU.load_benchmark_dataset used to build its few-shot example set only once,
from the FIRST loaded task's dev split. Every subsequent task was then
prompted with few-shot examples from the first task's subject (e.g. running
all 57 tasks prompted 56 subjects with the first task's dev examples instead
of their own). These tests are offline: datasets.load_dataset is mocked.
"""

from unittest import mock

import datasets

from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.schema import MultipleChoiceSchema


def _make_dataset(subject: str):
    dev = [
        {
            "question": f"[{subject}] dev question {i}",
            "choices": ["w", "x", "y", "z"],
            "answer": i % 4,
        }
        for i in range(5)
    ]
    test = [
        {
            "question": f"[{subject}] test question 0",
            "choices": ["w", "x", "y", "z"],
            "answer": 1,
        }
    ]
    return {"dev": dev, "test": test}


def _fake_load_dataset(name, config):
    assert name == "cais/mmlu"
    return _make_dataset(config)


class _CapturingModel:
    """Fake model that records the prompts it was given and answers "A"."""

    def __init__(self):
        self.prompts = []

    def get_model_name(self):
        return "fake"

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        return MultipleChoiceSchema(answer="A")

    def batch_generate(self, prompts, schemas=None):
        self.prompts.extend(prompts)
        return [schema(answer="A") for schema in schemas]


def _load_two_task_benchmark():
    t1, t2 = MMLUTask.ABSTRACT_ALGEBRA, MMLUTask.ANATOMY
    bench = MMLU(tasks=[t1, t2], n_shots=2)
    bench.load_benchmark_dataset(t1)
    goldens_t2 = bench.load_benchmark_dataset(t2)
    return bench, t2, goldens_t2


def _assert_prompt_uses_task_shots(prompt):
    # The anatomy prompt must contain anatomy's dev examples...
    assert "[anatomy] dev question 0" in prompt
    assert "[anatomy] dev question 1" in prompt
    # ...and not abstract_algebra's (the first task loaded).
    assert "[abstract_algebra]" not in prompt


def test_mmlu_predict_uses_each_tasks_own_few_shot_examples():
    with mock.patch.object(
        datasets, "load_dataset", side_effect=_fake_load_dataset
    ):
        bench, t2, goldens_t2 = _load_two_task_benchmark()
        model = _CapturingModel()
        bench.predict(model, t2, goldens_t2[0])

    _assert_prompt_uses_task_shots(model.prompts[0])


def test_mmlu_batch_predict_uses_each_tasks_own_few_shot_examples():
    with mock.patch.object(
        datasets, "load_dataset", side_effect=_fake_load_dataset
    ):
        bench, t2, goldens_t2 = _load_two_task_benchmark()
        model = _CapturingModel()
        bench.batch_predict(model, t2, goldens_t2)

    assert len(model.prompts) == len(goldens_t2)
    for prompt in model.prompts:
        _assert_prompt_uses_task_shots(prompt)
