import sys
from unittest.mock import MagicMock, patch

import pytest

try:
    import datasets  # noqa: F401
except ImportError:
    sys.modules["datasets"] = MagicMock()

from deepeval.benchmarks.gsm8k.gsm8k import GSM8K
from deepeval.benchmarks.schema import NumberSchema
from deepeval.dataset import Golden


@pytest.fixture
def goldens():
    return [
        Golden(input="q1", expected_output="5"),
        Golden(input="q2", expected_output="7"),
    ]


def test_gsm8k_batch_predict_batches_and_scores(goldens):
    benchmark = GSM8K(n_shots=0)
    benchmark.shots_dataset = []

    model = MagicMock()
    model.batch_generate.return_value = [
        NumberSchema(answer=5),
        NumberSchema(answer=7),
    ]

    with patch(
        "deepeval.benchmarks.gsm8k.gsm8k.GSM8KTemplate.generate_output",
        side_effect=lambda **kwargs: f"prompt::{kwargs['input']}",
    ):
        results = benchmark.batch_predict(model, goldens)

    model.batch_generate.assert_called_once()
    assert model.batch_generate.call_args.kwargs["prompts"] == [
        "prompt::q1",
        "prompt::q2",
    ]
    assert [r["prediction"] for r in results] == ["5", "7"]
    assert all(r["score"] for r in results)


def test_gsm8k_evaluate_routes_to_batch_when_batch_size_set(goldens):
    benchmark = GSM8K(n_shots=0, n_problems=2)
    model = MagicMock()

    with patch(
        "deepeval.benchmarks.gsm8k.gsm8k.capture_benchmark_run"
    ), patch.object(
        benchmark, "load_benchmark_dataset", return_value=goldens
    ), patch.object(
        benchmark,
        "batch_predict",
        return_value=[
            {"prediction": "5", "score": True},
            {"prediction": "7", "score": True},
        ],
    ) as mock_batch, patch.object(
        benchmark, "predict"
    ) as mock_predict:
        benchmark.evaluate(model, batch_size=2)

    mock_batch.assert_called_once()
    mock_predict.assert_not_called()
    assert benchmark.overall_score == 1.0
