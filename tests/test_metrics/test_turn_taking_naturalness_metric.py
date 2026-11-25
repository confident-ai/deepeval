import pytest

from deepeval.metrics import TurnTakingNaturalnessMetric
from deepeval.test_case import ConversationalTestCase
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    recording,
    CLEAN_GAPS,
    measure,
    speech_like,
    stereo_recording,
    timed_call,
)


def test_a_call_without_start_times_is_skipped():
    turns = timed_call(CLEAN_GAPS)
    turns[0].audio.start_time = None
    metric = measure(
        TurnTakingNaturalnessMetric(use_speech_detection=False), turns
    )
    assert metric.score is None
    assert metric.reason == (
        "Every audio turn needs a start_time to time the call, so none was "
        "inferred."
    )


def test_turn_stamps_are_named_as_the_timing_source():
    metric = measure(
        TurnTakingNaturalnessMetric(use_speech_detection=False),
        timed_call(CLEAN_GAPS),
    )
    assert metric.score_breakdown["timing_source"] == "turn_timestamps"


def test_a_caller_cutting_in_counts_against_the_reply_cut_into(tmp_path):
    # The caller talks over the first reply from 4 s to 6 s.
    path = stereo_recording(
        tmp_path / "call.wav",
        caller=[
            (0.0, speech_like(2.0)),
            (4.0, speech_like(2.0, seed=1)),
            (10.0, speech_like(2.0, seed=2)),
        ],
        agent=[
            (2.5, speech_like(3.0, seed=3)),
            (7.5, speech_like(2.0, seed=4)),
            (12.5, speech_like(2.0, seed=5)),
        ],
    )
    metric = TurnTakingNaturalnessMetric(use_speech_detection=False)
    metric.measure(
        ConversationalTestCase(
            turns=timed_call(CLEAN_GAPS), audio=recording(path)
        ),
        _show_indicator=False,
    )
    breakdown = metric.score_breakdown
    assert breakdown["timing_source"] == "call_recording"
    assert breakdown["eligible_turns"] == 3
    (event,) = breakdown["events"]
    assert event["metric"] == "User Interruption"
    assert event["caller_start"] == pytest.approx(4.0, abs=0.05)
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == (
        "1 of 3 agent replies had an issue: User Interruption (1)."
    )


def test_the_decision_model_checks_the_interruption_parts(tmp_path):
    path = stereo_recording(
        tmp_path / "call.wav",
        caller=[
            (0.0, speech_like(2.0)),
            (4.0, speech_like(2.0, seed=1)),
            (10.0, speech_like(2.0, seed=2)),
        ],
        agent=[
            (2.5, speech_like(3.0, seed=3)),
            (7.5, speech_like(2.0, seed=4)),
            (12.5, speech_like(2.0, seed=5)),
        ],
    )
    model = FakeAudioDecisionModel(probability=0.1)
    metric = TurnTakingNaturalnessMetric(
        use_speech_detection=False,
        eval_mode="hybrid",
        audio_decision_model=model,
    )
    metric.measure(
        ConversationalTestCase(
            turns=timed_call(CLEAN_GAPS), audio=recording(path)
        ),
        _show_indicator=False,
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert metric.reason.endswith(
        "1 flagged event was dismissed by the audio decision model."
    )
    assert metric.evaluation_cost == pytest.approx(0.001)
