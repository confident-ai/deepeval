import pytest

from deepeval.metrics.voice import (
    AbruptPitchJumpMetric,
    AgentResponsivenessMetric,
    AudioClippingMetric,
    AudioDropoutMetric,
    AudioGlitchMetric,
    AudioLoopMetric,
    BackgroundNoiseMetric,
    DeadStreamMetric,
    FailedResponseMetric,
    HeldToneMetric,
    LoudnessConsistencyMetric,
    LoudnessMetric,
    PausesMetric,
    PitchConsistencyMetric,
    PitchVariationMetric,
    RepromptMetric,
    SilentReplyMetric,
    SpeakingRateMetric,
    SpeechIntelligibilityMetric,
    UnexpectedCallEndMetric,
    VoiceNaturalnessMetric,
    VolumeVarianceMetric,
)
from deepeval.metrics.voice.utils import call_activity as activity_module
from deepeval.metrics.voice.utils import speech_detection
from deepeval.metrics.voice.utils.call_activity import (
    agent_replies,
    recording_replies,
)
from deepeval.metrics.voice.utils.defects import (
    CALL_RECORDING_UNREADABLE_REASON,
    STEREO_RECORDING_NEEDED_REASON,
    TURN_AUDIO_NEEDED_REASON,
)
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils
from tests.test_metrics.helpers import (
    RATE,
    agent_speech,
    clipped,
    other_voice,
    recording,
    speech_like,
    stereo_recording,
    with_gain,
)

TRANSCRIPT = [
    Turn(role="assistant", content="Hi, thanks for calling"),
    Turn(role="user", content="I need to change my booking."),
    Turn(role="assistant", content="Sure, what is the booking number"),
    Turn(role="user", content="It is four four seven one."),
    Turn(role="assistant", content="Thanks, that is changed now"),
]
# Agent replies at 0, 6 and 12 s with a caller line between each.
REPLY_STARTS = [0.0, 6.0, 12.0]


@pytest.fixture(autouse=True)
def _speech_model_hears_all_sound(monkeypatch):
    def probabilities(samples, sample_rate):
        count = int(
            len(samples) * speech_detection.SPEECH_MODEL_RATE / sample_rate
        ) // (speech_detection.SPEECH_WINDOW_SAMPLES)
        return [1.0] * count

    monkeypatch.setattr(speech_detection, "speech_probabilities", probabilities)
    activity_module._recording_cache.clear()
    yield
    activity_module._recording_cache.clear()


def _call(tmp_path, *, replies=None, caller=None, turns=TRANSCRIPT):
    path = stereo_recording(
        tmp_path / "call.wav",
        caller=caller
        or [(3.5, other_voice(2.0, seed=1)), (9.5, other_voice(2.0, seed=2))],
        agent=replies
        or [
            (start, speech_like(3.0, base_hz=230.0, seed=index))
            for index, start in enumerate(REPLY_STARTS)
        ],
    )
    return ConversationalTestCase(turns=turns, audio=recording(path))


def _measure(metric, test_case):
    metric.measure(test_case, _show_indicator=False)
    return metric


def _reply_starts(test_case):
    return [start for start, _ in recording_replies(test_case, True)]


def test_each_reply_runs_until_the_caller_speaks(tmp_path):
    starts = _reply_starts(_call(tmp_path))

    assert starts == pytest.approx(REPLY_STARTS, abs=0.05)


def test_the_agents_own_pause_stays_inside_its_reply(tmp_path):
    test_case = _call(
        tmp_path,
        replies=[(0.0, agent_speech(2.0)), (5.0, agent_speech(2.0))],
        caller=[(9.0, other_voice(2.0))],
    )

    replies = recording_replies(test_case, True)

    assert len(replies) == 1
    start, audio = replies[0]
    assert start == pytest.approx(0.0, abs=0.05)
    assert audio.duration == pytest.approx(7.0, abs=0.3)


def test_caller_sound_over_the_agent_does_not_split_its_reply(tmp_path):
    test_case = _call(
        tmp_path,
        replies=[(0.0, agent_speech(3.0)), (4.5, agent_speech(2.0))],
        caller=[(1.0, other_voice(1.0)), (8.0, other_voice(2.0))],
    )

    assert _reply_starts(test_case) == pytest.approx([0.0], abs=0.05)


def test_a_reply_is_cut_from_its_first_to_its_last_agent_sound(tmp_path):
    test_case = _call(
        tmp_path,
        replies=[(1.0, agent_speech(2.0))],
        caller=[(5.0, other_voice(2.0))],
    )

    ((start, audio),) = recording_replies(test_case, True)

    assert start == pytest.approx(1.0, abs=0.05)
    assert audio.start_time == start
    assert audio.duration == pytest.approx(2.0, abs=0.3)


def test_turn_audio_is_judged_instead_of_the_recording(tmp_path):
    test_case = _call(tmp_path)
    turns = [
        Turn(
            role=turn.role,
            content=turn.content,
            audio=Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(
                    agent_speech(3.0).tobytes(), RATE, num_channels=1
                ),
                "audio/wav",
            ),
        )
        for turn in TRANSCRIPT
    ]

    replies = agent_replies(test_case.model_copy(update={"turns": turns}))

    assert replies.by_turn
    assert sorted(replies.audio) == [0, 2, 4]


def test_a_mono_recording_has_no_replies(tmp_path):
    path = tmp_path / "mono.wav"
    path.write_bytes(
        audio_utils.pcm16_to_wav_bytes(
            agent_speech(3.0).tobytes(), RATE, num_channels=1
        )
    )
    test_case = ConversationalTestCase(turns=TRANSCRIPT, audio=recording(path))

    assert recording_replies(test_case, True) is None
    assert agent_replies(test_case).by_turn


@pytest.mark.parametrize(
    "metric",
    [
        AudioClippingMetric(),
        AudioDropoutMetric(),
        AudioLoopMetric(),
        HeldToneMetric(),
        DeadStreamMetric(),
        BackgroundNoiseMetric(),
        LoudnessMetric(),
        LoudnessConsistencyMetric(),
        AbruptPitchJumpMetric(),
        UnexpectedCallEndMetric(),
    ],
    ids=lambda metric: metric.__name__,
)
def test_a_clean_call_on_a_recording_alone_passes(tmp_path, metric):
    _measure(metric, _call(tmp_path))

    assert metric.score == 1.0
    assert not metric.score_breakdown["events"]


@pytest.mark.parametrize(
    "metric",
    [
        VolumeVarianceMetric(),
        PausesMetric(),
        PitchConsistencyMetric(),
        PitchVariationMetric(),
        SpeakingRateMetric(),
    ],
    ids=lambda metric: metric.__name__,
)
def test_a_recording_alone_reaches_the_reply_metrics(tmp_path, metric):
    _measure(metric, _call(tmp_path))

    reasons = [entry["error"] for entry in metric.score_breakdown["skipped"]]
    assert "no audio" not in metric.reason.lower()
    assert all(
        "reply_start" in entry for entry in metric.score_breakdown["skipped"]
    ), reasons


def test_a_clipped_reply_is_flagged_where_it_starts_on_the_call(tmp_path):
    test_case = _call(
        tmp_path,
        replies=[
            (0.0, speech_like(3.0, base_hz=230.0)),
            (6.0, clipped(3.0)),
            (12.0, speech_like(3.0, base_hz=230.0, seed=2)),
        ],
    )

    metric = _measure(AudioClippingMetric(), test_case)

    (event,) = metric.score_breakdown["events"]
    assert metric.score == pytest.approx(2 / 3)
    assert event["reply_start"] == pytest.approx(6.0, abs=0.05)
    assert "turn" not in event
    assert event["timing_source"] == "call_recording"
    assert event["start_time"] >= event["reply_start"]


def test_a_quiet_reply_is_placed_over_its_whole_reply(tmp_path):
    test_case = _call(
        tmp_path,
        replies=[
            (0.0, speech_like(3.0, base_hz=230.0)),
            (6.0, with_gain(speech_like(3.0, base_hz=230.0, seed=1), 0.15)),
            (12.0, speech_like(3.0, base_hz=230.0, seed=2)),
        ],
    )

    metric = _measure(LoudnessMetric(), test_case)

    (event,) = metric.score_breakdown["events"]
    assert event["type"] == "too_quiet"
    assert event["mark"] == "whole_reply"
    assert event["start_time"] == pytest.approx(event["reply_start"])
    assert metric.score_breakdown["timing_source"] == "call_recording"


def test_a_long_pause_inside_a_reply_is_flagged(tmp_path):
    test_case = _call(
        tmp_path,
        replies=[(0.0, agent_speech(3.0)), (8.0, agent_speech(3.0))],
        caller=[(13.0, other_voice(2.0))],
    )

    metric = _measure(PausesMetric(), test_case)

    (event,) = metric.score_breakdown["events"]
    assert event["type"] == "long_pause"
    assert event["reply_start"] == pytest.approx(0.0, abs=0.05)


def test_turn_audio_wins_over_the_recording(tmp_path):
    test_case = _call(tmp_path)
    turns = [
        Turn(
            role=turn.role,
            content=turn.content,
            audio=Audio.from_bytes(
                audio_utils.pcm16_to_wav_bytes(
                    with_gain(agent_speech(3.0), 0.1).tobytes(),
                    RATE,
                    num_channels=1,
                ),
                "audio/wav",
            ),
        )
        for turn in TRANSCRIPT
    ]

    metric = _measure(
        LoudnessMetric(), test_case.model_copy(update={"turns": turns})
    )

    assert [event["turn"] for event in metric.score_breakdown["events"]] == [
        0,
        2,
        4,
    ]
    assert "timing_source" not in metric.score_breakdown


@pytest.mark.parametrize(
    "metric",
    [FailedResponseMetric(), RepromptMetric(), SilentReplyMetric()],
    ids=lambda metric: metric.__name__,
)
def test_metrics_that_match_lines_to_audio_skip_a_recording_alone(
    tmp_path, metric
):
    _measure(metric, _call(tmp_path))

    assert metric.score is None
    assert metric.reason == TURN_AUDIO_NEEDED_REASON


def _mono_call(tmp_path):
    path = tmp_path / "mono.wav"
    path.write_bytes(
        audio_utils.pcm16_to_wav_bytes(
            agent_speech(3.0).tobytes(), RATE, num_channels=1
        )
    )
    return ConversationalTestCase(turns=TRANSCRIPT, audio=recording(path))


def test_a_mono_recording_alone_says_it_needs_stereo(tmp_path):
    metric = _measure(LoudnessMetric(), _mono_call(tmp_path))

    assert metric.score is None
    assert metric.reason == STEREO_RECORDING_NEEDED_REASON


def test_a_mono_recording_keeps_the_reason_of_metrics_that_need_turn_audio(
    tmp_path,
):
    metric = _measure(FailedResponseMetric(), _mono_call(tmp_path))

    assert metric.reason == TURN_AUDIO_NEEDED_REASON


def test_a_recording_that_cannot_be_read_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(activity_module, "_recording_bytes", lambda audio: None)

    metric = _measure(LoudnessMetric(), _call(tmp_path))

    assert metric.score is None
    assert metric.reason == CALL_RECORDING_UNREADABLE_REASON


def test_a_roll_up_without_speech_detection_never_runs_it(
    tmp_path, monkeypatch
):
    def unavailable(samples, sample_rate):
        raise ImportError("onnxruntime is not installed")

    monkeypatch.setattr(speech_detection, "speech_probabilities", unavailable)

    metric = _measure(
        AudioGlitchMetric(use_speech_detection=False), _call(tmp_path)
    )

    assert metric.score is not None


def test_a_roll_up_counts_each_recording_reply_once(tmp_path):
    test_case = _call(
        tmp_path,
        replies=[
            (0.0, speech_like(3.0, base_hz=230.0)),
            (6.0, clipped(3.0)),
            (12.0, speech_like(3.0, base_hz=230.0, seed=2)),
        ],
    )

    metric = _measure(SpeechIntelligibilityMetric(), test_case)

    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["turns_with_events"] == 1
    assert {
        round(event["reply_start"])
        for event in metric.score_breakdown["events"]
    } == {6}


def test_a_roll_up_of_prosody_runs_on_a_recording_alone(tmp_path):
    metric = _measure(VoiceNaturalnessMetric(), _call(tmp_path))

    assert metric.score is not None


def test_an_unexpected_end_is_placed_where_the_recording_ends(tmp_path):
    test_case = _call(tmp_path).model_copy(
        update={"metadata": {"end_reason": "ERROR"}}
    )

    metric = _measure(UnexpectedCallEndMetric(), test_case)

    (event,) = metric.score_breakdown["events"]
    assert metric.score == 0.0
    assert event["reply_start"] == pytest.approx(REPLY_STARTS[-1], abs=0.05)
    assert event["start_time"] == pytest.approx(15.0, abs=0.6)


def test_agent_responsiveness_scores_from_the_parts_that_can_run(tmp_path):
    metric = _measure(AgentResponsivenessMetric(), _call(tmp_path))

    assert metric.score == 1.0
    assert (
        metric.score_breakdown["metrics"]["Failed Response"]["reason"]
        == TURN_AUDIO_NEEDED_REASON
    )
