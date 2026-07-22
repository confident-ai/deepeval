import math
from types import SimpleNamespace

import pytest

from deepeval.errors import DeepEvalError
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import resolve_weighted_score
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams


class StubModel(DeepEvalBaseLLM):
    """Custom model without generate_raw_response / logprob support."""

    def __init__(self):
        super().__init__(model="stub-model")

    def load_model(self):
        return self

    def generate(self, *args, **kwargs) -> str:
        return '{"score": 8, "reason": "stub"}'

    async def a_generate(self, *args, **kwargs) -> str:
        return '{"score": 8, "reason": "stub"}'

    def get_model_name(self):
        return "stub-model"


def make_response_without_logprobs():
    message = SimpleNamespace(content='{"score": 8, "reason": "ok"}')
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=None, message=message)]
    )


def make_response_with_logprobs(score_token, top_logprobs):
    message = SimpleNamespace(content='{"score": 8, "reason": "ok"}')
    logprobs = SimpleNamespace(
        content=[SimpleNamespace(token=score_token, top_logprobs=top_logprobs)]
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=logprobs, message=message)]
    )


class TestResolveWeightedScore:
    def test_top_token_mode_returns_raw_score_without_touching_logprobs(self):
        response = make_response_without_logprobs()
        assert resolve_weighted_score(8, response, "top_token") == 8

    def test_logprobs_weighted_mode_raises_when_logprobs_missing(self):
        response = make_response_without_logprobs()
        with pytest.raises(DeepEvalError, match="top_token"):
            resolve_weighted_score(8, response, "logprobs_weighted")

    def test_auto_mode_falls_back_to_raw_score_with_warning(self):
        response = make_response_without_logprobs()
        with pytest.warns(UserWarning, match="top token"):
            score = resolve_weighted_score(8, response, "auto")
        assert score == 8

    def test_logprobs_weighted_mode_computes_weighted_score(self):
        response = make_response_with_logprobs(
            "8",
            [
                SimpleNamespace(token="8", logprob=math.log(0.6)),
                SimpleNamespace(token="9", logprob=math.log(0.4)),
            ],
        )
        score = resolve_weighted_score(8, response, "logprobs_weighted")
        assert score == pytest.approx(8.4)


class TestGEvalScoreMode:
    def make_test_case(self):
        return LLMTestCase(input="question", actual_output="answer")

    def make_metric(self, score_mode):
        return GEval(
            name="score mode test",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            evaluation_steps=["Check the answer addresses the question."],
            model=StubModel(),
            async_mode=False,
            score_mode=score_mode,
        )

    def test_rejects_invalid_score_mode(self):
        with pytest.raises(ValueError, match="score_mode"):
            self.make_metric("bogus")

    def test_logprobs_weighted_mode_raises_for_model_without_raw_response(
        self,
    ):
        metric = self.make_metric("logprobs_weighted")
        with pytest.raises(DeepEvalError, match="logprob"):
            metric.measure(
                self.make_test_case(),
                _show_indicator=False,
                _log_metric_to_confident=False,
            )
