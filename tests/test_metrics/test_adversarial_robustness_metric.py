"""Tests for AdversarialRobustnessMetric.

These tests use a fully mocked evaluation model (a `DeepEvalBaseLLM` test
double that returns canned schema instances) and a plain Python
`model_callback`, so they make NO real LLM/API calls and do not require
OPENAI_API_KEY.
"""

import asyncio
import pytest
from unittest.mock import patch

from deepeval.metrics.community import AdversarialRobustnessMetric
from deepeval.metrics.community.adversarial_robustness.schema import (
    Perturbation,
    Perturbations,
    RobustnessVerdict,
    Verdicts,
    AdversarialRobustnessScoreReason,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.errors import MissingTestCaseParamsError

_PERTURBATION_TYPES = ["semantic", "orthographic"]


class FakeJudgeModel(DeepEvalBaseLLM):
    """Returns canned schema instances based on the requested schema.

    Always reports ``num_perturbations`` perturbations and a matching list of
    verdicts where the first ``num_fragile`` are "no" (not robust) and the rest
    are "yes" (robust), so perturbations and verdicts always align.
    """

    def __init__(self, num_perturbations: int = 4, num_fragile: int = 1):
        self.num_perturbations = num_perturbations
        self.num_fragile = num_fragile
        super().__init__(model="fake-judge")

    def load_model(self):
        return None

    def get_model_name(self) -> str:
        return "fake-judge"

    def _respond(self, schema):
        if schema is Perturbations:
            perturbations = [
                Perturbation(
                    perturbed_input=f"perturbed input {i}",
                    perturbation_type=_PERTURBATION_TYPES[
                        i % len(_PERTURBATION_TYPES)
                    ],
                )
                for i in range(self.num_perturbations)
            ]
            return Perturbations(perturbations=perturbations)
        if schema is Verdicts:
            verdicts = [
                (
                    RobustnessVerdict(
                        verdict="no", reason=f"answer changed ({i})"
                    )
                    if i < self.num_fragile
                    else RobustnessVerdict(verdict="yes", reason=None)
                )
                for i in range(self.num_perturbations)
            ]
            return Verdicts(verdicts=verdicts)
        if schema is AdversarialRobustnessScoreReason:
            return AdversarialRobustnessScoreReason(
                reason="canned reason for testing."
            )
        raise AssertionError(f"unexpected schema requested: {schema}")

    def generate(self, prompt: str, schema=None, *args, **kwargs):
        return self._respond(schema)

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs):
        return self._respond(schema)


class RecordingCallback:
    """Sync system-under-test stub that records the inputs it was probed with."""

    def __init__(self, response: str = "Paris"):
        self.response = response
        self.calls = []

    def __call__(self, perturbed_input: str) -> str:
        self.calls.append(perturbed_input)
        return self.response


def make_metric(
    callback,
    *,
    num_perturbations: int = 4,
    num_fragile: int = 1,
    async_mode: bool = False,
    strict_mode: bool = False,
    threshold: float = 0.5,
) -> AdversarialRobustnessMetric:
    with patch(
        "deepeval.metrics.community.adversarial_robustness.adversarial_robustness.initialize_model"
    ) as mock_init:
        mock_init.return_value = (
            FakeJudgeModel(num_perturbations, num_fragile),
            False,
        )
        return AdversarialRobustnessMetric(
            model_callback=callback,
            threshold=threshold,
            num_perturbations=num_perturbations,
            async_mode=async_mode,
            strict_mode=strict_mode,
        )


def make_test_case() -> LLMTestCase:
    return LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
    )


def test_sync_measure_scores_fraction_of_robust_perturbations():
    callback = RecordingCallback()
    metric = make_metric(
        callback, num_perturbations=4, num_fragile=1, async_mode=False
    )
    score = metric.measure(make_test_case(), _show_indicator=False)

    # 3 of 4 perturbations were robust -> 0.75
    assert score == pytest.approx(0.75)
    assert metric.success is True
    assert metric.reason is not None
    assert len(metric.perturbations) == 4
    assert len(metric.verdicts) == 4
    # The system under test must be probed once per perturbation.
    assert len(callback.calls) == 4
    assert metric.is_successful() is True


def test_async_measure_via_default_async_mode():
    callback = RecordingCallback()
    metric = make_metric(
        callback, num_perturbations=4, num_fragile=1, async_mode=True
    )
    score = metric.measure(make_test_case(), _show_indicator=False)

    assert score == pytest.approx(0.75)
    assert len(callback.calls) == 4
    assert metric.success is True


def test_a_measure_directly():
    callback = RecordingCallback()
    metric = make_metric(
        callback, num_perturbations=2, num_fragile=0, async_mode=True
    )
    score = asyncio.run(
        metric.a_measure(make_test_case(), _show_indicator=False)
    )

    # Fully robust -> 1.0
    assert score == pytest.approx(1.0)
    assert metric.success is True
    assert len(callback.calls) == 2


def test_non_robust_model_fails_threshold():
    callback = RecordingCallback()
    metric = make_metric(
        callback, num_perturbations=4, num_fragile=4, async_mode=False
    )
    score = metric.measure(make_test_case(), _show_indicator=False)

    assert score == pytest.approx(0.0)
    assert metric.success is False
    assert metric.is_successful() is False


def test_async_callback_is_awaited():
    """An async `model_callback` should be awaited in the async path."""
    seen = []

    async def async_callback(perturbed_input: str) -> str:
        seen.append(perturbed_input)
        return "Paris"

    metric = make_metric(
        async_callback, num_perturbations=3, num_fragile=0, async_mode=True
    )
    score = metric.measure(make_test_case(), _show_indicator=False)

    assert score == pytest.approx(1.0)
    assert len(seen) == 3


def test_strict_mode_zeroes_out_imperfect_score():
    callback = RecordingCallback()
    metric = make_metric(
        callback, num_perturbations=4, num_fragile=1, async_mode=False
    )
    # strict_mode forces threshold to 1; recompute under strict.
    metric.strict_mode = True
    metric.threshold = 1
    score = metric.measure(make_test_case(), _show_indicator=False)

    assert score == pytest.approx(0.0)
    assert metric.success is False


def test_invalid_model_callback_raises():
    with pytest.raises(ValueError):
        with patch(
            "deepeval.metrics.community.adversarial_robustness.adversarial_robustness.initialize_model"
        ) as mock_init:
            mock_init.return_value = (FakeJudgeModel(), False)
            AdversarialRobustnessMetric(model_callback="not-callable")


def test_invalid_perturbation_types_raises():
    with pytest.raises(ValueError):
        with patch(
            "deepeval.metrics.community.adversarial_robustness.adversarial_robustness.initialize_model"
        ) as mock_init:
            mock_init.return_value = (FakeJudgeModel(), False)
            AdversarialRobustnessMetric(
                model_callback=lambda x: "out",
                perturbation_types=["semantic", "phonetic"],
            )


def test_missing_actual_output_raises():
    callback = RecordingCallback()
    metric = make_metric(callback, async_mode=False)
    tc = LLMTestCase(input="What is the capital of France?", actual_output=None)
    with pytest.raises(MissingTestCaseParamsError):
        metric.measure(tc, _show_indicator=False)
