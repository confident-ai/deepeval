"""Offline regression coverage for async conversational window ordering."""

import asyncio
from contextlib import nullcontext
import os

os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"

import pytest

from deepeval.metrics import TurnFaithfulnessMetric
from deepeval.metrics.turn_faithfulness import turn_faithfulness as module
from deepeval.metrics.turn_faithfulness.schema import (
    InteractionFaithfulnessScore,
)
from deepeval.test_case import ConversationalTestCase, Turn


def make_case():
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="first user"),
            Turn(
                role="assistant",
                content="first answer",
                retrieval_context=["first fact"],
            ),
            Turn(role="user", content="second user"),
            Turn(
                role="assistant",
                content="second answer",
                retrieval_context=["second fact"],
            ),
        ]
    )


def make_three_window_case():
    turns = []
    for index in range(3):
        turns.extend(
            [
                Turn(role="user", content=f"user {index}"),
                Turn(
                    role="assistant",
                    content=f"answer {index}",
                    retrieval_context=[f"fact {index}"],
                ),
            ]
        )
    return ConversationalTestCase(turns=turns)


def make_metric(async_mode=True):
    metric = TurnFaithfulnessMetric.__new__(TurnFaithfulnessMetric)
    metric.threshold = 0.5
    metric.model = None
    metric.using_native_model = False
    metric.evaluation_model = "offline stub"
    metric.window_size = 1
    metric.async_mode = async_mode
    metric.include_reason = True
    metric.error = None
    return metric


def stub_system_one(monkeypatch):
    async def no_system_one(*args, **kwargs):
        return False

    monkeypatch.setattr(module, "a_run_system_one_eval", no_system_one)
    monkeypatch.setattr(module, "run_system_one_eval", lambda *a, **kw: False)
    monkeypatch.setattr(
        module,
        "metric_progress_indicator",
        lambda *args, **kwargs: nullcontext(),
    )


def interaction_score(is_first):
    return InteractionFaithfulnessScore(
        score=0.2 if is_first else 0.8,
        reason="first" if is_first else "second",
        claims=[],
        truths=[],
        verdicts=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_delay,second_delay",
    [(0.02, 0), (0, 0.02)],
    ids=["first-finishes-last", "first-finishes-first"],
)
async def test_async_windows_keep_conversation_order(
    monkeypatch, first_delay, second_delay
):
    metric = make_metric()
    stub_system_one(monkeypatch)

    async def fake_window_scores(turns_window, multimodal):
        is_first = turns_window[0].content == "first user"
        await asyncio.sleep(first_delay if is_first else second_delay)
        return [interaction_score(is_first)]

    monkeypatch.setattr(
        metric, "_a_get_faithfulness_scores", fake_window_scores
    )

    observed_reason_order = []

    async def fake_final_reason(scores):
        observed_reason_order.extend(score.reason for score in scores)
        return "stub"

    monkeypatch.setattr(metric, "_a_generate_reason", fake_final_reason)

    score = await metric.a_measure(make_case(), _show_indicator=False)

    assert score == pytest.approx(0.5)
    assert observed_reason_order == ["first", "second"]
    assert "Reason: first" in metric.verbose_logs.split("Window 2")[0]
    assert "Reason: second" in metric.verbose_logs.split("Window 2")[1]


@pytest.mark.asyncio
async def test_async_three_windows_preserve_batches_when_finishing_in_reverse(
    monkeypatch,
):
    metric = make_metric()
    metric.window_size = 1
    stub_system_one(monkeypatch)

    all_started = asyncio.Event()
    release = [asyncio.Event() for _ in range(3)]
    completed = [asyncio.Event() for _ in range(3)]
    finish_order = []
    batches = {
        0: [
            InteractionFaithfulnessScore(
                score=0.0, reason="0a", claims=[], truths=[], verdicts=[]
            ),
            InteractionFaithfulnessScore(
                score=0.2, reason="0b", claims=[], truths=[], verdicts=[]
            ),
        ],
        1: [
            InteractionFaithfulnessScore(
                score=0.4, reason="1a", claims=[], truths=[], verdicts=[]
            ),
            InteractionFaithfulnessScore(
                score=0.6, reason="1b", claims=[], truths=[], verdicts=[]
            ),
        ],
        2: [
            InteractionFaithfulnessScore(
                score=0.8, reason="2a", claims=[], truths=[], verdicts=[]
            ),
            InteractionFaithfulnessScore(
                score=1.0, reason="2b", claims=[], truths=[], verdicts=[]
            ),
        ],
    }
    started = set()

    async def fake_window_scores(turns_window, multimodal):
        index = int(turns_window[0].content.removeprefix("user "))
        started.add(index)
        if len(started) == 3:
            all_started.set()
        await all_started.wait()
        await release[index].wait()
        finish_order.append(index)
        completed[index].set()
        return batches[index]

    monkeypatch.setattr(
        metric, "_a_get_faithfulness_scores", fake_window_scores
    )
    observed_reason_order = []

    async def fake_final_reason(scores):
        observed_reason_order.extend(score.reason for score in scores)
        return "stub"

    monkeypatch.setattr(metric, "_a_generate_reason", fake_final_reason)
    measuring = asyncio.create_task(
        metric.a_measure(make_three_window_case(), _show_indicator=False)
    )
    await asyncio.wait_for(all_started.wait(), timeout=1)
    for index in (2, 1, 0):
        release[index].set()
        await asyncio.wait_for(completed[index].wait(), timeout=1)

    score = await measuring

    assert finish_order == [2, 1, 0]
    assert observed_reason_order == ["0a", "0b", "1a", "1b", "2a", "2b"]
    assert score == pytest.approx(0.5)
    logged_reasons = [
        section.split("Reason: ", 1)[1].split()[0]
        for section in metric.verbose_logs.split("Window ")[1:]
    ]
    assert logged_reasons == ["0a", "0b", "1a", "1b", "2a", "2b"]


@pytest.mark.asyncio
async def test_async_empty_window_set_keeps_baseline_score(monkeypatch):
    metric = make_metric()
    stub_system_one(monkeypatch)

    async def no_window_scores(*args, **kwargs):
        raise AssertionError("no windows should be scored")

    async def stub_reason(scores):
        assert scores == []
        return "stub"

    monkeypatch.setattr(metric, "_a_get_faithfulness_scores", no_window_scores)
    monkeypatch.setattr(metric, "_a_generate_reason", stub_reason)
    case = ConversationalTestCase(
        turns=[Turn(role="user", content="unanswered", retrieval_context=[])]
    )

    score = await metric.a_measure(case, _show_indicator=False)

    assert score == pytest.approx(1.0)
    assert metric.reason == "stub"


def test_empty_turns_keep_validation_error():
    with pytest.raises(TypeError, match="turns.*must not be empty"):
        ConversationalTestCase(turns=[])


@pytest.mark.asyncio
async def test_async_window_exception_propagates(monkeypatch):
    metric = make_metric()
    stub_system_one(monkeypatch)
    failure = RuntimeError("offline window failure")

    async def fail_window_scores(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        metric, "_a_get_faithfulness_scores", fail_window_scores
    )

    with pytest.raises(RuntimeError, match="offline window failure") as raised:
        await metric.a_measure(make_case(), _show_indicator=False)

    assert raised.value is failure


@pytest.mark.asyncio
async def test_async_system_one_short_circuit_does_not_score_windows(
    monkeypatch,
):
    metric = make_metric()
    metric.score = 0.75

    async def system_one_available(*args, **kwargs):
        return True

    async def no_window_scores(*args, **kwargs):
        raise AssertionError("system-one result should short-circuit windows")

    monkeypatch.setattr(module, "a_run_system_one_eval", system_one_available)
    monkeypatch.setattr(metric, "_a_get_faithfulness_scores", no_window_scores)

    score = await metric.a_measure(make_case(), _show_indicator=False)

    assert score == pytest.approx(0.75)


def test_sync_windows_keep_conversation_order(monkeypatch):
    metric = make_metric(async_mode=False)
    stub_system_one(monkeypatch)

    def fake_window_scores(turns_window, multimodal):
        is_first = turns_window[0].content == "first user"
        return [interaction_score(is_first)]

    monkeypatch.setattr(metric, "_get_faithfulness_scores", fake_window_scores)

    observed_reason_order = []

    def fake_final_reason(scores):
        observed_reason_order.extend(score.reason for score in scores)
        return "stub"

    monkeypatch.setattr(metric, "_generate_reason", fake_final_reason)

    score = metric.measure(make_case(), _show_indicator=False)

    assert score == pytest.approx(0.5)
    assert observed_reason_order == ["first", "second"]
    assert "Reason: first" in metric.verbose_logs.split("Window 2")[0]
    assert "Reason: second" in metric.verbose_logs.split("Window 2")[1]
