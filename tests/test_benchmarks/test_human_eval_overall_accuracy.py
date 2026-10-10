"""Regression test: HumanEval's "Overall Accuracy" must be the mean pass@k
over tasks, not the fraction of tasks with pass@k > 0.

`predict()` returns `score` as the pass@k estimate for a task -- a float in
[0, 1], not a pass/fail flag. `evaluate()` previously did `if score:
task_correct = 1`, so any task with even one correct sample among n
generations (score > 0) counted as a full pass. Two fractional scores that
are individually below 0.5 (0.3 and 0.4) are used below so a binarized
"count each other 1" bug and the real mean (0.35) are unambiguously
different numbers -- this can't pass by coincidence.

The benchmark's own dataset loading and code-sample generation are bypassed
(`__new__` + monkeypatching `predict`/`load_benchmark_dataset`, the same
pattern `test_bbh_batch_predict_scores_schema_answer_correctly` uses for
BigBenchHard) so this stays a fast, offline unit test.
"""

import contextlib

import pytest

import deepeval.benchmarks.human_eval.human_eval as human_eval_module
from deepeval.benchmarks.human_eval.human_eval import HumanEval
from deepeval.benchmarks.human_eval.task import HumanEvalTask
from deepeval.dataset import Golden
from deepeval.scorer import Scorer


def _human_eval(tasks):
    bench = HumanEval.__new__(HumanEval)
    bench.tasks = tasks
    bench.n = 10
    bench.temperature = 0.0
    bench.verbose_mode = False
    bench.scorer = Scorer()
    bench.c = {}
    bench.functions = {}
    return bench


def test_overall_accuracy_is_the_mean_pass_at_k(monkeypatch):
    tasks = [
        HumanEvalTask.HAS_CLOSE_ELEMENTS,
        HumanEvalTask.SEPARATE_PAREN_GROUPS,
    ]
    per_task_score = {
        HumanEvalTask.HAS_CLOSE_ELEMENTS: 0.3,
        HumanEvalTask.SEPARATE_PAREN_GROUPS: 0.4,
    }
    bench = _human_eval(tasks)

    monkeypatch.setattr(
        human_eval_module,
        "capture_benchmark_run",
        lambda *a, **k: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        bench,
        "load_benchmark_dataset",
        lambda task: Golden(input="x", expected_output="y"),
    )
    monkeypatch.setattr(
        bench,
        "predict",
        lambda model, task, golden, k: {
            "prediction": "def f(): ...",
            "score": per_task_score[task],
        },
    )

    bench.evaluate(model=object(), k=1)

    assert bench.overall_score == pytest.approx(0.35)
    assert bench.task_scores["Score"].tolist() == pytest.approx([0.3, 0.4])
