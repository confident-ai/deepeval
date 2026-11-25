import pytest

from deepeval.metrics import AbruptCutoffMetric
from deepeval.test_case import ConversationalTestCase
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    recording,
    cut_short,
    measure,
    speech_like,
    stereo_recording,
    timed_call,
    with_defect,
    with_noise,
)


def test_a_reply_ending_mid_sound_is_an_abrupt_cut_off_without_a_timeline():
    turns = with_defect(1, lambda: cut_short(speech_like(3.0), 1.125))
    for turn in turns:
        turn.audio.start_time = None
    metric = measure(AbruptCutoffMetric(), turns)
    (event,) = metric.score_breakdown["events"]
    assert event["type"] == "abrupt_cutoff"
    assert event["turn"] == 1
    assert event["mark"] == "moment"
    assert event["duration_seconds"] == pytest.approx(0.02)
    assert event["offset_seconds"] == pytest.approx(1.105, abs=0.01)


def test_a_reply_the_caller_barged_into_is_not_an_abrupt_cut_off():
    turns = with_defect(1, lambda: cut_short(speech_like(3.0), 1.125))
    turns[1].interrupted = True
    assert measure(AbruptCutoffMetric(), turns).score == 1.0


def test_a_reply_ending_in_steady_background_noise_is_not_a_cut_off():
    turns = with_defect(1, lambda: with_noise(speech_like(3.0), -30))
    assert measure(AbruptCutoffMetric(), turns).score == 1.0


def test_a_reply_cut_mid_word_over_background_noise_is_a_cut_off():
    turns = with_defect(
        1, lambda: cut_short(with_noise(speech_like(4.0), -30), 2.125)
    )
    assert measure(AbruptCutoffMetric(), turns).score == pytest.approx(2 / 3)


def _recorded_cutoff(
    tmp_path, *, agent, caller=(), seconds=None, audio_decision_model=None
):
    metric = AbruptCutoffMetric(
        use_speech_detection=False,
        eval_mode="hybrid" if audio_decision_model else None,
        audio_decision_model=audio_decision_model,
    )
    path = stereo_recording(
        tmp_path / "call.wav", caller=caller, agent=agent, seconds=seconds
    )
    metric.measure(
        ConversationalTestCase(turns=timed_call([0.5]), audio=recording(path)),
        _show_indicator=False,
    )
    return metric


def test_a_reply_cut_mid_sound_on_a_recording_is_a_cutoff(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[
            (0.0, speech_like(2.0)),
            (3.0, cut_short(speech_like(3.0), 1.125)),
        ],
        caller=[(6.0, speech_like(1.0))],
    )
    assert metric.score == 0.5
    assert metric.score_breakdown["timing_source"] == "call_recording"
    [event] = metric.score_breakdown["events"]
    assert event["reply_start"] == pytest.approx(3.0, abs=0.05)
    assert event["stopped_at_seconds"] == pytest.approx(4.125, abs=0.05)
    # The 20 ms the cut was judged on, ending where the sound stops.
    assert event["start_time"] + event["duration_seconds"] == pytest.approx(
        event["stopped_at_seconds"], abs=0.01
    )
    assert event["mark"] == "moment"
    assert event["timing_source"] == "call_recording"


def test_an_agent_cut_off_by_the_caller_speaking_is_excused(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[
            (0.0, speech_like(2.0)),
            (3.0, cut_short(speech_like(3.0), 1.125)),
        ],
        caller=[(3.6, speech_like(2.0))],
    )
    assert metric.score == 1.0


def test_a_reply_that_fades_out_on_a_recording_is_not_a_cutoff(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[(0.0, speech_like(2.0)), (3.0, speech_like(2.0))],
        caller=[(6.0, speech_like(1.0))],
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_sound_running_to_the_end_of_the_recording_is_not_judged(tmp_path):
    metric = _recorded_cutoff(
        tmp_path,
        agent=[
            (0.0, speech_like(2.0)),
            (3.0, cut_short(speech_like(3.0), 1.125)),
        ],
        seconds=4.125,
    )
    assert metric.score == 1.0


def test_a_cutoff_the_decision_model_hears_as_a_clean_ending_is_dismissed(
    tmp_path,
):
    model = FakeAudioDecisionModel(probability=0.1)
    metric = _recorded_cutoff(
        tmp_path,
        agent=[
            (0.0, speech_like(2.0)),
            (3.0, cut_short(speech_like(3.0), 1.125)),
        ],
        caller=[(6.0, speech_like(1.0))],
        audio_decision_model=model,
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert len(model.clips) == 1
