import pytest

from deepeval.metrics import (
    ConversationalToolCorrectnessMetric,
    SimulationEfficiencyMetric,
    SimulationPathMetric,
    TurnTakingNaturalnessMetric,
)
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn
from tests.test_metrics.test_voice_metrics import make_tone


def _call(name, **arguments):
    return ToolCall(name=name, input_parameters=arguments)


def _conversation(*tools, metadata=None):
    return ConversationalTestCase(
        turns=[
            Turn(role="user", content="Stop my card."),
            Turn(role="assistant", content="Done.", tools_called=list(tools)),
        ],
        metadata=metadata,
    )


def _measure(metric, test_case):
    metric.measure(test_case, _show_indicator=False)
    return metric


def test_tool_calls_match_on_listed_arguments_only():
    metric = _measure(
        ConversationalToolCorrectnessMetric(
            expected_tools=[_call("freeze_card", last_four=4417)]
        ),
        _conversation(_call("freeze_card", last_four="4417", reason="fraud")),
    )
    assert metric.score == 1.0


def test_a_missing_tool_call_lowers_the_score():
    metric = _measure(
        ConversationalToolCorrectnessMetric(
            expected_tools=[
                _call("freeze_card", last_four="4417"),
                _call("order_replacement_card", last_four="4417"),
            ]
        ),
        _conversation(_call("freeze_card", last_four="4417")),
    )
    assert metric.score == 0.5
    assert metric.score_breakdown["missing"] == [
        "order_replacement_card(last_four='4417')"
    ]
    assert not metric.is_successful()


def test_a_wrong_argument_does_not_match():
    metric = _measure(
        ConversationalToolCorrectnessMetric(
            expected_tools=[_call("freeze_card", last_four="4417")]
        ),
        _conversation(_call("freeze_card", last_four="4471")),
    )
    assert metric.score == 0.0


def test_a_forbidden_call_fails_even_when_every_expected_call_was_made():
    metric = _measure(
        ConversationalToolCorrectnessMetric(
            expected_tools=[_call("freeze_card", last_four="4417")],
            forbidden_tools=[_call("raise_dispute", amounts=["18.60"])],
        ),
        _conversation(
            _call("freeze_card", last_four="4417"),
            _call("raise_dispute", amounts=["312.40", "59.99", "18.60"]),
        ),
    )
    assert metric.score == 0.0
    assert metric.score_breakdown["forbidden_called"]


def test_ordering_is_only_checked_when_asked_for():
    calls = (_call("raise_dispute"), _call("freeze_card"))
    expected = [_call("freeze_card"), _call("raise_dispute")]
    unordered = _measure(
        ConversationalToolCorrectnessMetric(expected_tools=expected),
        _conversation(*calls),
    )
    ordered = _measure(
        ConversationalToolCorrectnessMetric(
            expected_tools=expected, should_consider_ordering=True
        ),
        _conversation(*calls),
    )
    assert unordered.score == 1.0
    assert ordered.score == 0.5
    assert ordered.score_breakdown["out_of_order"] == ["raise_dispute()"]


def _path(ending, failed=(), path=("open", "done"), shortest=2):
    return {
        "Simulation path": list(path),
        "Simulation ending": ending,
        "Failed nodes": list(failed),
        "Shortest path": shortest,
    }


def test_path_passes_on_an_ending_without_failed_nodes():
    metric = _measure(
        SimulationPathMetric(), _conversation(metadata=_path("resolved"))
    )
    assert metric.score == 1.0


@pytest.mark.parametrize(
    "metadata",
    [
        _path(None),
        _path("off_script"),
        _path("resolved", failed=["skipped_code"]),
    ],
)
def test_path_fails_without_an_ending_off_script_or_through_a_failed_node(
    metadata,
):
    metric = _measure(SimulationPathMetric(), _conversation(metadata=metadata))
    assert metric.score == 0.0


def test_path_is_skipped_without_a_simulation_graph():
    metric = _measure(SimulationPathMetric(), _conversation())
    assert metric.score is None


def test_efficiency_is_shortest_path_over_user_turns():
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="a"),
            Turn(role="assistant", content="b"),
            Turn(role="user", content="c"),
            Turn(role="assistant", content="d"),
            Turn(role="user", content="e"),
            Turn(role="assistant", content="f"),
            Turn(role="user", content="g"),
            Turn(role="assistant", content="h"),
        ],
        metadata=_path("resolved", shortest=2),
    )
    metric = _measure(SimulationEfficiencyMetric(), test_case)
    assert metric.score == 0.5
    assert metric.score_breakdown["extra_user_turns"] == 2


def _spoken(role, start, seconds, content="Okay.", **kwargs):
    return Turn(
        role=role,
        content=content,
        audio=make_tone(start_time=start, duration=seconds),
        **kwargs,
    )


def _sub_score(test_case, name):
    metric = _measure(TurnTakingNaturalnessMetric(sub_scores=[name]), test_case)
    return metric.score_breakdown["sub_scores"][name]


def test_unknown_sub_score_is_rejected():
    with pytest.raises(ValueError):
        TurnTakingNaturalnessMetric(sub_scores=["politeness"])


def test_talking_into_a_mid_sentence_pause_fails_that_pause():
    pause = {
        "events": [
            {"type": "MidSentencePause", "offset_seconds": 1.0, "seconds": 2.0}
        ]
    }
    waited = ConversationalTestCase(
        turns=[
            _spoken("user", 0.0, 4.0, metadata=pause),
            _spoken("assistant", 4.5, 1.0),
        ]
    )
    talked = ConversationalTestCase(
        turns=[
            _spoken("user", 0.0, 4.0, metadata=pause),
            _spoken("assistant", 1.5, 1.0),
        ]
    )
    assert _sub_score(waited, "mid_utterance_pauses")["score"] == 1.0
    assert _sub_score(talked, "mid_utterance_pauses")["score"] == 0.0


def test_a_sub_score_is_none_when_the_call_never_tested_it():
    test_case = ConversationalTestCase(
        turns=[_spoken("user", 0.0, 1.0), _spoken("assistant", 1.5, 1.0)]
    )
    assert _sub_score(test_case, "mid_utterance_pauses")["score"] is None
    assert _sub_score(test_case, "dead_air")["score"] is None


def test_stopping_for_a_backchannel_fails():
    heard = {"backchannels": [{"content": "Mm-hmm.", "offset_seconds": 1.0}]}
    kept_going = ConversationalTestCase(
        turns=[
            _spoken("user", 0.0, 1.0),
            _spoken("assistant", 1.5, 4.0, metadata=heard),
        ]
    )
    stopped = ConversationalTestCase(
        turns=[
            _spoken("user", 0.0, 1.0),
            _spoken(
                "assistant",
                1.5,
                1.3,
                content="Your policy covers",
                metadata=heard,
            ),
        ]
    )
    assert _sub_score(kept_going, "backchannel_vs_barge_in")["score"] == 1.0
    assert _sub_score(stopped, "backchannel_vs_barge_in")["score"] == 0.0


def test_a_long_silence_before_a_tool_reply_fails_tool_latency():
    def reply_after(seconds):
        return ConversationalTestCase(
            turns=[
                _spoken("user", 0.0, 1.0),
                _spoken(
                    "assistant",
                    1.0 + seconds,
                    1.0,
                    tools_called=[_call("check_order_status")],
                ),
            ]
        )

    assert _sub_score(reply_after(1.0), "tool_latency")["score"] == 1.0
    assert _sub_score(reply_after(6.0), "tool_latency")["score"] == 0.0


def test_hanging_up_during_a_hold_fails():
    hold = {
        "events": [{"type": "Hold", "offset_seconds": 1.0, "seconds": 20.0}]
    }
    waited = ConversationalTestCase(
        turns=[
            _spoken("user", 0.0, 23.0, metadata=hold),
            _spoken("assistant", 10.0, 1.0, content="Take your time."),
            _spoken("assistant", 23.5, 1.0),
        ]
    )
    hung_up = ConversationalTestCase(
        turns=[
            _spoken("assistant", 0.0, 1.0),
            _spoken("user", 1.5, 23.0, metadata=hold),
        ],
        metadata={"Stop reason": "The agent ended the call"},
    )
    assert _sub_score(waited, "waiting_through_hold")["score"] == 1.0
    assert _sub_score(hung_up, "waiting_through_hold")["score"] == 0.0


def test_an_agent_that_cuts_itself_off_fails_echo():
    test_case = ConversationalTestCase(
        turns=[
            _spoken("user", 0.0, 1.0),
            _spoken("assistant", 1.5, 1.0, interrupted=True),
            _spoken("user", 4.0, 1.0),
            _spoken("assistant", 5.5, 1.0),
        ]
    )
    assert _sub_score(test_case, "echo")["score"] == 0.5


def test_voice_style_scores_the_agents_audio_with_an_audio_judge():
    from deepeval.metrics import VoiceStyleMetric
    from deepeval.metrics.voice_style.schema import VoiceStyleVerdict
    from deepeval.models import DeepEvalBaseLLM

    class _Listener(DeepEvalBaseLLM):
        def __init__(self):
            self.heard = []
            super().__init__(model="listener")

        def load_model(self):
            return self

        def generate(self, prompt, schema=None, audio=None):
            self.heard.append(audio)
            return VoiceStyleVerdict(
                score=8, reason="Slow and warm throughout."
            )

        async def a_generate(self, prompt, schema=None, audio=None):
            return self.generate(prompt, schema, audio)

        def supports_audio(self):
            return True

        def get_model_name(self):
            return "listener"

    listener = _Listener()
    test_case = ConversationalTestCase(
        turns=[_spoken("user", 0.0, 1.0), _spoken("assistant", 1.5, 1.0)]
    )
    metric = VoiceStyleMetric(style="Speak slowly and warmly.", model=listener)
    metric.measure(test_case, _show_indicator=False)

    assert metric.score == 0.8
    assert metric.reason == "Slow and warm throughout."
    assert len(listener.heard[0]) == 1
