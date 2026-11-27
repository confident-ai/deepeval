import pytest

from deepeval.metrics import ResponseDelayConsistencyMetric
from tests.test_metrics.helpers import (
    measure,
    measure_recording,
    speech_like,
    timed_call,
)


def test_an_instant_reply_among_slow_ones_is_inconsistent():
    turns = timed_call([3.0, 3.0, 0.0, 3.0])
    turns[5].latency_ms = 0.0
    [event] = measure(
        ResponseDelayConsistencyMetric(use_speech_detection=False), turns
    ).score_breakdown["events"]
    assert event["turn"] == 5
    # The other waits are heard from where the caller's speech ends, a
    # quarter second before their clip does.
    assert event["shift_seconds"] == pytest.approx(-3.25, abs=0.05)


def test_delay_consistency_needs_three_replies():
    metric = measure(
        ResponseDelayConsistencyMetric(use_speech_detection=False),
        timed_call([0.5, 4.0]),
    )
    assert metric.score is None
    assert metric.reason == (
        "At least 3 timed assistant replies are needed to compare their "
        "delays."
    )


def test_delay_consistency_reads_the_same_waits_from_a_recording(tmp_path):
    agent = [(0.0, speech_like(1.0))]
    caller = []
    cursor = 1.5
    for wait in (0.5, 0.5, 4.0, 0.5):
        caller.append((cursor, speech_like(1.0)))
        agent.append((cursor + 0.75 + wait, speech_like(1.0)))
        cursor += 0.75 + wait + 2.0
    metric = measure_recording(
        ResponseDelayConsistencyMetric(use_speech_detection=False),
        tmp_path,
        caller=caller,
        agent=agent,
    )
    assert metric.score == 0.75
    [event] = metric.score_breakdown["events"]
    assert event["shift_seconds"] == pytest.approx(3.5, abs=0.05)
