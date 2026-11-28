import pytest

from deepeval.metrics import SpeakingRateMetric
from deepeval.metrics.voice.prosody.speaking_rate.speaking_rate import (
    spells_something_out,
)
from deepeval.metrics.voice.utils.analysis import syllable_rate
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    RATE,
    clean_turns,
    edit_reply,
    measure,
    set_content,
    syllables,
    with_defect,
    with_reply_audio,
)


def _events(samples, **limits):
    metric = measure(
        SpeakingRateMetric(**limits), with_defect(1, lambda: samples)
    )
    return metric.score_breakdown["events"]


@pytest.mark.parametrize("per_second", [3, 5, 8])
def test_syllables_are_counted_from_the_audio(per_second):
    rate = syllable_rate(syllables(per_second, 4.0), RATE)
    assert rate == pytest.approx(per_second, rel=0.15)


def test_an_ordinary_pace_is_not_flagged():
    assert _events(syllables(5, 4.0)) == []


def test_a_rushed_reply_is_too_fast():
    [event] = _events(syllables(12, 4.0))
    assert event["type"] == "too_fast"
    assert event["turn"] == 1
    assert "syllables a second" in event["description"]


def test_a_dragged_out_reply_is_too_slow():
    [event] = _events(syllables(2.5, 6.0))
    assert event["type"] == "too_slow"


def test_stopping_mid_reply_is_not_speaking_slowly():
    assert _events(syllables(5, 4.0, pause_after=2.0, pause=3.0)) == []


def test_speaking_rate_limits_are_configurable():
    samples = syllables(5, 4.0)
    [event] = _events(samples, max_syllables_per_second=4.5)
    assert event["type"] == "too_fast"
    with pytest.raises(ValueError):
        SpeakingRateMetric(
            min_syllables_per_second=8.0, max_syllables_per_second=4.0
        )


def test_a_reply_with_too_little_speech_is_skipped_for_speaking_rate():
    metric = measure(
        SpeakingRateMetric(),
        with_reply_audio(1, syllables(5, 1.0), clean_turns(rhythm=2.5)),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert metric.score_breakdown["skipped"] == [
        {"turn": 1, "error": "Reply has under 2 seconds of speech."}
    ]


def test_the_words_of_a_reply_do_not_change_its_speaking_rate():
    turns = with_defect(1, lambda: syllables(5, 4.0))
    turns[1].content = " ".join(["word"] * 60)
    assert measure(SpeakingRateMetric(), turns).score == 1.0


def _slow_reply(content):
    turns = with_reply_audio(1, syllables(2.5, 6.0), clean_turns(rhythm=2.5))
    return edit_reply(1, set_content(content), turns)


def test_a_slow_reply_that_spells_something_out_is_skipped():
    metric = measure(
        SpeakingRateMetric(), _slow_reply("B, Y, R, O, N. Is that correct?")
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert metric.score_breakdown["events"] == []
    assert metric.score_breakdown["skipped"] == [
        {"turn": 1, "error": "Reply spells something out."}
    ]


def test_the_same_slow_reply_in_ordinary_words_is_too_slow():
    metric = measure(
        SpeakingRateMetric(), _slow_reply("Byron. Is that correct?")
    )
    assert metric.score == pytest.approx(2 / 3)
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "too_slow"
    assert metric.score_breakdown["skipped"] == []


@pytest.mark.parametrize(
    "content,spelled_out",
    [
        ("B, Y, R, O, N. Is that correct?", True),
        ("That's B-Y-R-O-N", True),
        ("My number is 5 5 5 0 1 4 2", True),
        ("A, B, C", True),
        ("A, B", False),
        ("B as in Boy", True),
        ("Your order SH5216742 has shipped", True),
        ("At 9 a.m. I'd say so", False),
        ("I have a question for you.", False),
        ("That's at 2 p.m. today", False),
        ("Call 555-0142", False),
        ("", False),
        (None, False),
    ],
)
def test_letters_digits_and_codes_read_out_spell_something_out(
    content, spelled_out
):
    assert spells_something_out(content) is spelled_out


def test_a_slightly_slow_reply_is_only_flagged_with_a_decision_model():
    assert _events(syllables(4.5, 4.0)) == []
    model = FakeAudioDecisionModel(probability=0.9)
    [event] = _events(
        syllables(4.5, 4.0), eval_mode="hybrid", audio_decision_model=model
    )
    assert event["type"] == "too_slow"
    assert model.questions == ["Is the speaker talking unnaturally slowly?"]


def test_a_slightly_fast_reply_is_asked_whether_it_is_too_fast():
    model = FakeAudioDecisionModel(probability=0.9)
    [event] = _events(
        syllables(9, 4.0), eval_mode="hybrid", audio_decision_model=model
    )
    assert event["type"] == "too_fast"
    assert model.questions == ["Is the speaker talking unnaturally fast?"]


def test_a_pace_the_decision_model_finds_natural_is_dismissed():
    model = FakeAudioDecisionModel(probability=0.1)
    metric = measure(
        SpeakingRateMetric(eval_mode="hybrid", audio_decision_model=model),
        with_defect(1, lambda: syllables(4.5, 4.0)),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1


def test_an_ordinary_pace_is_never_sent_to_the_decision_model():
    model = FakeAudioDecisionModel(probability=0.9)
    assert (
        _events(
            syllables(6.5, 4.0), eval_mode="hybrid", audio_decision_model=model
        )
        == []
    )
    assert model.clips == []
