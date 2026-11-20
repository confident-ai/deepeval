"""
Offline tests for the HaluEval benchmark: golden construction, answer
normalization and scoring. No model, network, dataset download, or API key
required.
"""

import pytest

from deepeval.benchmarks import HaluEval
from deepeval.benchmarks.tasks import HaluEvalTask
from deepeval.benchmarks.halu_eval.template import HaluEvalTemplate
from deepeval.benchmarks.schema import AffirmationSchema

QA_ROW = {
    "knowledge": "Canberra is the capital city of Australia.",
    "question": "What is the capital of Australia?",
    "right_answer": "Canberra",
    "hallucinated_answer": "Sydney is the capital of Australia.",
}
DIALOGUE_ROW = {
    "knowledge": "Iron Man is starring Robert Downey Jr.",
    "dialogue_history": "[Human]: Who stars in Iron Man?",
    "right_response": "Robert Downey Jr. stars in it.",
    "hallucinated_response": "Tom Cruise stars in it.",
}
SUMMARIZATION_ROW = {
    "document": "The council approved a new park on Monday.",
    "right_summary": "A new park was approved.",
    "hallucinated_summary": "A new stadium was approved by the mayor.",
}


@pytest.mark.parametrize(
    "task,row,right,hallucinated",
    [
        (HaluEvalTask.QA, QA_ROW, "Canberra", "Sydney"),
        (HaluEvalTask.DIALOGUE, DIALOGUE_ROW, "Robert", "Tom Cruise"),
        (HaluEvalTask.SUMMARIZATION, SUMMARIZATION_ROW, "park", "stadium"),
    ],
)
def test_paired_tasks_yield_right_then_hallucinated(
    task, row, right, hallucinated
):
    samples = HaluEvalTemplate.format_samples(row, task)
    assert [expected for _, expected in samples] == ["No", "Yes"]
    assert right in samples[0][0]
    assert hallucinated in samples[1][0]
    assert hallucinated not in samples[0][0]
    for input, _ in samples:
        assert "Answer 'Yes' or 'No'" in input


@pytest.mark.parametrize(
    "label,expected", [("yes", "Yes"), ("no", "No"), (" Yes ", "Yes")]
)
def test_general_task_uses_human_label(label, expected):
    row = {
        "ID": "1",
        "user_query": "Name a prime number.",
        "chatgpt_response": "4",
        "hallucination": label,
        "hallucination_spans": [],
    }
    samples = HaluEvalTemplate.format_samples(row, HaluEvalTask.GENERAL)
    assert len(samples) == 1
    assert samples[0][1] == expected


@pytest.mark.parametrize(
    "raw,normalized",
    [
        ("Yes", "Yes"),
        ("yes.", "Yes"),
        ("  No, the answer is supported.", "No"),
        ("**Yes**", "Yes"),
        ("Maybe", "Maybe"),
        ("", ""),
    ],
)
def test_normalize_prediction(raw, normalized):
    assert HaluEvalTemplate.normalize_prediction(raw) == normalized


class _FakeModel:
    """Says 'Yes' when the prompt mentions Sydney, otherwise 'No'."""

    def __init__(self, use_schema=True):
        self.use_schema = use_schema

    def get_model_name(self):
        return "fake"

    def _answer(self, prompt):
        return "Yes" if "Sydney" in prompt else "No"

    def generate(self, prompt, schema=None):
        if schema is None:
            return self._answer(prompt) + "."
        if not self.use_schema:
            raise TypeError("schema not supported")
        return schema(answer=self._answer(prompt))

    def batch_generate(self, prompts, schemas=None):
        if schemas is None:
            return [self._answer(p) for p in prompts]
        if not self.use_schema:
            raise TypeError("schema not supported")
        return [s(answer=self._answer(p)) for p, s in zip(prompts, schemas)]


@pytest.fixture
def benchmark():
    benchmark = HaluEval(tasks=[HaluEvalTask.QA])
    # Skip the Hugging Face download: inject a one-row dataset.
    benchmark.datasets[HaluEvalTask.QA] = [QA_ROW]
    return benchmark


@pytest.mark.parametrize("use_schema", [True, False])
@pytest.mark.parametrize("batch_size", [None, 2])
def test_evaluate_scores_perfect_model(benchmark, use_schema, batch_size):
    result = benchmark.evaluate(
        model=_FakeModel(use_schema=use_schema), batch_size=batch_size
    )
    assert result.overall_accuracy == 1.0
    assert list(benchmark.predictions["Prediction"]) == ["No", "Yes"]
    assert list(benchmark.task_scores["Score"]) == [1.0]


def test_n_problems_per_task_limits_goldens(benchmark):
    benchmark.n_problems_per_task = 1
    benchmark.evaluate(model=_FakeModel())
    assert len(benchmark.predictions) == 1


def test_affirmation_schema_only_allows_yes_no():
    assert AffirmationSchema(answer="Yes").answer == "Yes"
    with pytest.raises(Exception):
        AffirmationSchema(answer="Maybe")
