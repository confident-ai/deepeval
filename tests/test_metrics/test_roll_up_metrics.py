import pytest

from deepeval.metrics import (
    AgentResponsivenessMetric,
    AudioGlitchMetric,
    AudioIntegrityMetric,
    SpeechIntelligibilityMetric,
    TurnTakingNaturalnessMetric,
    VoiceConsistencyMetric,
    VoiceNaturalnessMetric,
    VoiceReliabilityMetric,
)
from deepeval.test_case import ConversationalTestCase
from tests.test_metrics.helpers import (
    CLEAN_GAPS,
    STAMPS_NOTE,
    agent_speech,
    caller_turn,
    clean_turns,
    clipped,
    loop_clip,
    measure,
    speech_like,
    timed_call,
    tone,
    with_gain,
    with_noise,
    with_reply_audio,
)


def _reprompted_call():
    turns = clean_turns()
    turns.insert(3, caller_turn("Hello? Can you hear me?"))
    return turns


# Roll-ups that find speech inside each turn's audio; tests judge by loudness.
SPEECH_DETECTING = (
    AgentResponsivenessMetric,
    TurnTakingNaturalnessMetric,
    VoiceReliabilityMetric,
)


def _metric(metric_class, **kwargs):
    if metric_class in SPEECH_DETECTING:
        kwargs["use_speech_detection"] = False
    return metric_class(**kwargs)


# (metric, clean call, call with one issue, the part that flags it, what is
# counted on the clean call)
ROLL_UPS = [
    (
        VoiceNaturalnessMetric,
        clean_turns,
        lambda: with_reply_audio(
            3, speech_like(1.0) + clipped(0.5) + speech_like(1.0)
        ),
        "Audio Clipping",
        3,
    ),
    (
        SpeechIntelligibilityMetric,
        clean_turns,
        lambda: with_reply_audio(3, with_noise(agent_speech(), -30)),
        "Background Noise",
        3,
    ),
    (
        VoiceConsistencyMetric,
        clean_turns,
        lambda: with_reply_audio(3, with_gain(agent_speech(), 0.3)),
        "Loudness Consistency",
        3,
    ),
    (
        AudioIntegrityMetric,
        clean_turns,
        lambda: with_reply_audio(3, loop_clip()),
        "Audio Loop",
        3,
    ),
    (
        AudioGlitchMetric,
        clean_turns,
        lambda: with_reply_audio(
            3, speech_like(2.0) + tone(1.5, 388.0) + speech_like(1.0)
        ),
        "Held Tone",
        3,
    ),
    (
        AgentResponsivenessMetric,
        clean_turns,
        _reprompted_call,
        "Reprompt",
        3,
    ),
    (
        VoiceReliabilityMetric,
        clean_turns,
        lambda: with_reply_audio(3, loop_clip()),
        "Audio Loop",
        3,
    ),
    (
        TurnTakingNaturalnessMetric,
        lambda: timed_call(CLEAN_GAPS),
        lambda: timed_call([0.5, -1.5, 0.5]),
        "Agent Interruption",
        3,
    ),
]


def _reason(metric_class, reason):
    if metric_class is TurnTakingNaturalnessMetric:
        return reason + STAMPS_NOTE
    return reason


def _counted(metric_class):
    if metric_class in (AgentResponsivenessMetric, VoiceReliabilityMetric):
        return "exchange", "exchanges"
    return "agent reply", "agent replies"


@pytest.mark.parametrize("metric_class,clean,_,__,eligible", ROLL_UPS)
def test_a_clean_call_scores_one(metric_class, clean, _, __, eligible):
    metric = measure(_metric(metric_class), clean())
    assert metric.score == 1.0
    assert metric.reason == _reason(
        metric_class, f"No {_counted(metric_class)[0]} had an issue."
    )
    assert metric.score_breakdown["eligible_turns"] == eligible
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,_,with_issue,part,__", ROLL_UPS)
def test_an_issue_counts_against_its_reply(
    metric_class, _, with_issue, part, __
):
    metric = measure(_metric(metric_class), with_issue())
    breakdown = metric.score_breakdown
    flagged, eligible = (
        breakdown["turns_with_events"],
        breakdown["eligible_turns"],
    )
    assert flagged == 1
    assert metric.score == pytest.approx(1 - 1 / eligible)
    assert metric.reason == _reason(
        metric_class,
        f"1 of {eligible} {_counted(metric_class)[1]} had an issue: "
        f"{part} (1).",
    )
    assert {event["metric"] for event in breakdown["events"]} == {part}
    assert breakdown["metrics"][part]["turns_with_events"] == 1


@pytest.mark.parametrize("metric_class,_,with_issue,__,___", ROLL_UPS)
def test_strict_mode_zeroes_an_imperfect_score(
    metric_class, _, with_issue, __, ___
):
    metric = measure(_metric(metric_class, strict_mode=True), with_issue())
    assert metric.score == 0.0
    assert metric.success is False


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_class,_,with_issue,__,___", ROLL_UPS)
async def test_sync_and_async_agree(metric_class, _, with_issue, __, ___):
    test_case = ConversationalTestCase(turns=with_issue())
    sync_metric = _metric(metric_class)
    sync_metric.measure(test_case, _show_indicator=False)
    async_metric = _metric(metric_class)
    await async_metric.a_measure(test_case, _show_indicator=False)
    assert async_metric.score == sync_metric.score
    assert async_metric.score_breakdown == sync_metric.score_breakdown
    assert async_metric.reason == sync_metric.reason


@pytest.mark.parametrize("metric_class,clean,_,__,___", ROLL_UPS)
def test_the_breakdown_lists_every_part(metric_class, clean, _, __, ___):
    metric = _metric(metric_class)
    measure(metric, clean())
    assert list(metric.score_breakdown["metrics"]) == [
        part.__name__ for part in metric._parts()
    ]
