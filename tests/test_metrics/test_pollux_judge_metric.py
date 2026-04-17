import os

import pytest

from deepeval.metrics import PolluxJudgeMetric
from deepeval.metrics.pollux import POLLUX_TAGGED_FEEDBACK_RE, POLLUX_TAGGED_SCORE_RE
from deepeval.test_case import LLMTestCase


def _build_sync_client(response_text: str):
    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return _Response(response_text)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def _build_async_client(response_text: str):
    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        @staticmethod
        async def create(**kwargs):
            return _Response(response_text)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def _test_case() -> LLMTestCase:
    return LLMTestCase(
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        expected_output="4",
    )


class TestPolluxJudgeMetric:
    def test_sync_measure_normalized_score(self, monkeypatch):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
            async_mode=False,
        )
        monkeypatch.setattr(
            metric,
            "_get_sync_client",
            lambda: _build_sync_client("2"),
        )

        score = metric.measure(_test_case())

        assert score == 1.0
        assert metric.reason == ""
        assert metric.is_successful()

    def test_sync_measure_include_reason_false(self, monkeypatch):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
            async_mode=False,
            include_reason=False,
        )
        monkeypatch.setattr(
            metric,
            "_get_sync_client",
            lambda: _build_sync_client("2"),
        )

        metric.measure(_test_case())
        assert metric.reason is None

    def test_tagged_judge_output_with_patterns(self, monkeypatch):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
            async_mode=False,
            score_pattern=POLLUX_TAGGED_SCORE_RE,
            feedback_pattern=POLLUX_TAGGED_FEEDBACK_RE,
        )
        monkeypatch.setattr(
            metric,
            "_get_sync_client",
            lambda: _build_sync_client(
                "[FEEDBACK] good answer [RESULT] 2 [END]"
            ),
        )

        score = metric.measure(_test_case())

        assert score == 1.0
        assert metric.reason == "good answer"

    def test_non_zero_based_rubric_normalization(self, monkeypatch):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={1: "Bad", 2: "Okay", 3: "Great"},
            async_mode=False,
        )
        monkeypatch.setattr(
            metric,
            "_get_sync_client",
            lambda: _build_sync_client("2"),
        )

        score = metric.measure(_test_case())

        assert score == 0.5
        assert metric.reason == ""

    def test_normalize_score_false_returns_raw(self, monkeypatch):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
            normalize_score=False,
            async_mode=False,
        )
        monkeypatch.setattr(
            metric,
            "_get_sync_client",
            lambda: _build_sync_client("2"),
        )

        score = metric.measure(_test_case())

        assert score == 2.0

    def test_strict_mode_threshold_for_normalized(self):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
            strict_mode=True,
            normalize_score=True,
        )
        assert metric.threshold == 1.0

    def test_strict_mode_threshold_for_raw(self):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
            strict_mode=True,
            normalize_score=False,
        )
        assert metric.threshold == 2.0

    def test_parse_error_sets_error_and_raises(self, monkeypatch):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
            async_mode=False,
        )
        monkeypatch.setattr(
            metric,
            "_get_sync_client",
            lambda: _build_sync_client("no tags here"),
        )

        with pytest.raises(ValueError):
            metric.measure(_test_case())

        assert metric.error is not None
        assert "Failed to parse score" in metric.error

    def test_rubrics_validation(self):
        with pytest.raises(ValueError):
            PolluxJudgeMetric(
                criteria_name="Correctness",
                rubrics={0: "Only one level"},
            )

        with pytest.raises(ValueError):
            PolluxJudgeMetric(
                criteria_name="Correctness",
                rubrics={"bad": "Wrong", 1: "Right"},
            )

    @pytest.mark.asyncio
    async def test_async_measure(self, monkeypatch):
        metric = PolluxJudgeMetric(
            criteria_name="Correctness",
            rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
        )
        monkeypatch.setattr(
            metric,
            "_get_async_client",
            lambda: _build_async_client("1"),
        )

        score = await metric.a_measure(_test_case())

        assert score == 0.5
        assert metric.reason == ""


@pytest.mark.skipif(
    os.getenv("POLLUX_BASE_URL") is None,
    reason="POLLUX_BASE_URL is not set",
)
def test_integration_with_real_endpoint():
    kwargs = {}
    if os.getenv("POLLUX_USE_TAGGED_JUDGE_OUTPUT", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        kwargs["score_pattern"] = POLLUX_TAGGED_SCORE_RE
        kwargs["feedback_pattern"] = POLLUX_TAGGED_FEEDBACK_RE

    metric = PolluxJudgeMetric(
        criteria_name="Correctness",
        rubrics={0: "Wrong", 1: "Partial", 2: "Correct"},
        judge_model=os.getenv("POLLUX_MODEL", "ai-forever/Pollux-4B-Judge"),
        base_url=os.getenv("POLLUX_BASE_URL"),
        api_key=os.getenv("POLLUX_API_KEY", "NONE"),
        async_mode=False,
        **kwargs,
    )

    score = metric.measure(
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
        )
    )

    assert score is not None
