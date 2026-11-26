import json
import math

import pytest
from pydantic import ValidationError

from deepeval.metrics import (
    ImageCoherenceMetric,
    ImageEditingMetric,
    ImageHelpfulnessMetric,
    ImageReferenceMetric,
    TextToImageMetric,
)
from deepeval.metrics.multimodal_metrics.image_coherence.schema import (
    ReasonScore as ImageCoherenceReasonScore,
)
from deepeval.metrics.multimodal_metrics.image_editing.schema import (
    ReasonScore as ImageEditingReasonScore,
)
from deepeval.metrics.multimodal_metrics.image_helpfulness.schema import (
    ReasonScore as ImageHelpfulnessReasonScore,
)
from deepeval.metrics.multimodal_metrics.image_reference.schema import (
    ReasonScore as ImageReferenceReasonScore,
)
from deepeval.metrics.multimodal_metrics.text_to_image.schema import (
    ReasonScore as TextToImageReasonScore,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class _ScriptedImageJudge(DeepEvalBaseLLM):
    def __init__(self, scores):
        self.scores = list(scores)
        self.i = 0

    def load_model(self, *args, **kwargs):
        return None

    def _next(self):
        value = self.scores[min(self.i, len(self.scores) - 1)]
        self.i += 1
        return value

    def generate(self, prompt, schema=None, **kwargs):
        name = schema.__name__ if schema is not None else ""
        if name == "ReasonScore":
            return json.dumps({"score": self._next(), "reasoning": "stub"})
        return json.dumps({"reason": "stub"})

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return "scripted-image-judge"

    def supports_multimodal(self):
        return True


IMG_IN = "[DEEPEVAL:IMAGE:https://example.com/in.png]"
IMG_OUT = "[DEEPEVAL:IMAGE:https://example.com/out.png]"


@pytest.mark.parametrize(
    "schema_cls",
    [
        ImageCoherenceReasonScore,
        ImageHelpfulnessReasonScore,
        ImageReferenceReasonScore,
    ],
)
@pytest.mark.parametrize("bad_score", [-0.1, -5, 10.1, 25])
def test_scalar_image_reason_score_rejects_out_of_bounds(schema_cls, bad_score):
    with pytest.raises(ValidationError):
        schema_cls(reasoning="stub", score=bad_score)


@pytest.mark.parametrize(
    "schema_cls", [ImageEditingReasonScore, TextToImageReasonScore]
)
@pytest.mark.parametrize("bad_scores", [[], [-1], [10.5], [8, -2]])
def test_list_image_reason_score_rejects_invalid_scores(schema_cls, bad_scores):
    with pytest.raises(ValidationError):
        schema_cls(reasoning="stub", score=bad_scores)


@pytest.mark.parametrize(
    "metric_cls",
    [ImageCoherenceMetric, ImageHelpfulnessMetric, ImageReferenceMetric],
)
@pytest.mark.parametrize(
    ("raw_score", "expected_score"),
    [(25, 1.0), (-5, 0.0), (8, 0.8)],
)
@pytest.mark.parametrize("async_mode", [False, True])
async def test_scalar_image_metrics_clamp_scores(
    metric_cls, raw_score, expected_score, async_mode
):
    tc = LLMTestCase(input="caption", actual_output=f"text {IMG_OUT}")
    metric = metric_cls(
        model=_ScriptedImageJudge([raw_score]),
        async_mode=async_mode,
    )

    score = (
        await metric.a_measure(tc, _show_indicator=False)
        if async_mode
        else metric.measure(tc, _show_indicator=False)
    )

    assert score == pytest.approx(expected_score)
    assert metric.score == pytest.approx(expected_score)


@pytest.mark.parametrize(
    ("raw_scores", "expected_score"),
    [
        ([[-5], [-5]], 0.0),
        ([[10], [-5]], 0.0),
        ([[25], [25]], 1.0),
        ([[10], [4]], math.sqrt(40) / 10),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
async def test_image_editing_clamps_scores_without_domain_error(
    raw_scores, expected_score, async_mode
):
    tc = LLMTestCase(input=IMG_IN, actual_output=IMG_OUT)
    metric = ImageEditingMetric(
        model=_ScriptedImageJudge(raw_scores),
        async_mode=async_mode,
    )

    score = (
        await metric.a_measure(tc, _show_indicator=False)
        if async_mode
        else metric.measure(tc, _show_indicator=False)
    )

    assert score == pytest.approx(expected_score)
    assert metric.score == pytest.approx(expected_score)
    assert all(0.0 <= s <= 10.0 for s in metric.SC_scores + metric.PQ_scores)


@pytest.mark.parametrize(
    ("raw_scores", "expected_score"),
    [
        ([[-5], [-5]], 0.0),
        ([[10], [-5]], 0.0),
        ([[25], [25]], 1.0),
        ([[10], [4]], math.sqrt(40) / 10),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
async def test_text_to_image_clamps_scores_without_domain_error(
    raw_scores, expected_score, async_mode
):
    tc = LLMTestCase(input="draw a cat", actual_output=IMG_OUT)
    metric = TextToImageMetric(
        model=_ScriptedImageJudge(raw_scores),
        async_mode=async_mode,
    )

    score = (
        await metric.a_measure(tc, _show_indicator=False)
        if async_mode
        else metric.measure(tc, _show_indicator=False)
    )

    assert score == pytest.approx(expected_score)
    assert metric.score == pytest.approx(expected_score)
    assert all(0.0 <= s <= 10.0 for s in metric.SC_scores + metric.PQ_scores)
