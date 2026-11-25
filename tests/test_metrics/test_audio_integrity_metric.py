import pytest

from deepeval.metrics import AudioIntegrityMetric
from deepeval.test_case import ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    recording,
    clean_turns,
    cut_short,
    measure,
    speech_like,
    stereo_recording,
    timed_call,
    to_audio,
    undecodable,
    without_place,
)


def test_undecodable_agent_audio_fails_the_call():
    turns = clean_turns()
    undecodable(turns[3])
    metric = measure(AudioIntegrityMetric(), turns)
    assert metric.score == 0.0
    assert metric.score_breakdown["critical_failure"] is True
    (event,) = metric.score_breakdown["events"]
    assert event["type"] == "audio_undecodable"
    assert event["turn"] == 3


def test_a_call_without_agent_turns_fails():
    metric = measure(
        AudioIntegrityMetric(),
        [Turn(role="user", content="Hello?", audio=to_audio(speech_like(1.0)))],
    )
    assert metric.score == 0.0
    assert (
        metric.reason == "A critical failure occurred: assistant_turn_missing."
    )


def test_a_reply_with_no_audio_counts_against_it():
    turns = clean_turns()
    turns[3].audio = None
    metric = measure(AudioIntegrityMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)
    (event,) = metric.score_breakdown["events"]
    assert without_place(event) == {
        "type": "no_audio",
        "turn": 3,
        "metric": "Silent Reply",
    }


def _recording_with_a_cut(tmp_path):
    # Replies start at 2.5, 7.5 and 12.5 s; the middle one is cut mid-word.
    return stereo_recording(
        tmp_path / "call.wav",
        caller=[(0.0, speech_like(2.0)), (5.0, speech_like(2.0, seed=1))],
        agent=[
            (2.5, speech_like(2.0, seed=2)),
            (7.5, cut_short(speech_like(3.0, seed=3), 1.125)),
            (12.5, speech_like(2.0, seed=4)),
        ],
    )


def test_a_cut_off_on_the_recording_counts_against_the_reply_playing(
    tmp_path,
):
    test_case = ConversationalTestCase(
        turns=timed_call([0.5, 0.5, 0.5]),
        audio=recording(_recording_with_a_cut(tmp_path)),
    )
    metric = AudioIntegrityMetric(use_speech_detection=False)
    metric.measure(test_case, _show_indicator=False)
    (event,) = metric.score_breakdown["events"]
    assert event["metric"] == "Abrupt Cut-off"
    assert event["reply_start"] == 7.5
    assert metric.score == pytest.approx(2 / 3)


def test_cut_offs_in_turns_left_out_of_the_test_case_are_ignored(tmp_path):
    test_case = ConversationalTestCase(
        turns=timed_call([0.5, 0.5, 0.5])[-2:],
        audio=recording(_recording_with_a_cut(tmp_path)),
    )
    metric = AudioIntegrityMetric(use_speech_detection=False)
    metric.measure(test_case, _show_indicator=False)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1
    assert metric.score_breakdown["events"] == []
    assert metric.score_breakdown["metrics"]["Abrupt Cut-off"]["score"] == 1.0


def test_without_reply_start_times_cut_offs_come_from_each_reply(tmp_path):
    turns = timed_call([0.5, 0.5, 0.5])
    for turn in turns:
        turn.audio.start_time = None
    test_case = ConversationalTestCase(
        turns=turns, audio=recording(_recording_with_a_cut(tmp_path))
    )
    metric = AudioIntegrityMetric(use_speech_detection=False)
    metric.measure(test_case, _show_indicator=False)
    assert metric.score == 1.0
