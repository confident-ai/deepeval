"""Deterministic tests for TranscriptionAccuracyMetric.

A scripted judge stands in for the LLM, so none of this needs an API key.
What is pinned down here is everything the judge does not decide: which turns
get paired into an exchange, what reaches the prompt, the score arithmetic,
and the refusal to run on a conversation that carries no transcription at all.
"""

import pytest

from deepeval.metrics import TranscriptionAccuracyMetric
from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.transcription_accuracy.schema import (
    TranscriptionAccuracyScoreReason,
    TranscriptionAccuracyVerdict,
    Verdicts,
)
from deepeval.metrics.base_metric import Verdict
from deepeval.metrics.transcription_accuracy.transcription_accuracy import (
    VERBATIM_TRANSCRIPTION_PROMPT,
    _verdict_for_audio_exchange,
    get_audio_exchanges,
    get_transcribed_exchanges,
)
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseSTT, OpenAISTTModel
from deepeval.test_case import (
    Audio,
    ConversationalTestCase,
    MultiTurnParams,
    Turn,
)
from deepeval.voice.connectors import audio_utils


class ScriptedJudge(DeepEvalBaseLLM):
    """Returns one scripted verdict per exchange and records its prompts."""

    def __init__(self, verdicts):
        self.scripted = list(verdicts)
        self.prompts = []
        super().__init__(model="scripted-judge")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        if schema is TranscriptionAccuracyScoreReason:
            return TranscriptionAccuracyScoreReason(reason="scripted reason")
        return Verdicts(
            verdicts=[
                TranscriptionAccuracyVerdict(
                    verdict=verdict, reason="scripted reason"
                )
                for verdict in self.scripted
            ]
        )

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "scripted-judge"


def _measure(turns, scripted_verdicts, **metric_kwargs):
    judge = ScriptedJudge(scripted_verdicts)
    metric = TranscriptionAccuracyMetric(
        model=judge, async_mode=False, **metric_kwargs
    )
    metric.measure(ConversationalTestCase(turns=turns))
    return metric, judge


# ------------------------------------------------------------------ pairing


def test_a_caller_turn_pairs_with_the_reply_that_transcribed_it():
    exchanges = get_transcribed_exchanges(
        [
            Turn(role="user", content="I can start in two weeks"),
            Turn(
                role="assistant",
                content="Two weeks works.",
                provider_transcript="I can start in two months",
            ),
        ]
    )

    assert exchanges == [
        {
            "spoken": "I can start in two weeks",
            "transcribed": "I can start in two months",
            "agent_reply": "Two weeks works.",
        }
    ]


def test_consecutive_caller_turns_are_one_exchange():
    """A barge-in appends a second user turn before the agent answers, and the
    transcription covers everything the agent heard since it last spoke."""
    exchanges = get_transcribed_exchanges(
        [
            Turn(role="user", content="Hold on"),
            Turn(role="user", content="I meant two weeks"),
            Turn(
                role="assistant",
                content="Understood.",
                provider_transcript="hold on i meant two weeks",
            ),
        ]
    )

    assert len(exchanges) == 1
    assert exchanges[0]["spoken"] == "Hold on I meant two weeks"


def test_a_reply_without_a_transcription_takes_its_caller_turn_with_it():
    """That exchange is unmeasurable, so it must not be scored against a
    later reply's transcription."""
    exchanges = get_transcribed_exchanges(
        [
            Turn(role="user", content="first question"),
            Turn(role="assistant", content="unmeasured reply"),
            Turn(role="user", content="second question"),
            Turn(
                role="assistant",
                content="measured reply",
                provider_transcript="second question",
            ),
        ]
    )

    assert [exchange["spoken"] for exchange in exchanges] == ["second question"]


def test_an_agent_that_speaks_first_opens_no_exchange():
    exchanges = get_transcribed_exchanges(
        [
            Turn(
                role="assistant",
                content="Hello, how can I help?",
                provider_transcript="",
            ),
            Turn(role="user", content="hi"),
        ]
    )

    assert exchanges == []


def test_an_empty_transcription_is_still_judged():
    """Hearing nothing is a transcription failure, not a missing field."""
    exchanges = get_transcribed_exchanges(
        [
            Turn(role="user", content="I can start in two weeks"),
            Turn(role="assistant", content="Sorry?", provider_transcript=""),
        ]
    )

    assert len(exchanges) == 1
    assert exchanges[0]["transcribed"] == ""


# ------------------------------------------------------------------ scoring


def test_the_score_is_the_share_of_faithfully_heard_turns():
    turns = []
    for index in range(4):
        turns.append(Turn(role="user", content=f"question {index}"))
        turns.append(
            Turn(
                role="assistant",
                content=f"answer {index}",
                provider_transcript=f"question {index}",
            )
        )

    metric, _ = _measure(turns, ["yes", "no", "yes", "yes"])

    assert metric.score == 0.75
    assert metric.success is True


def test_a_conversation_heard_perfectly_scores_one():
    metric, _ = _measure(
        [
            Turn(role="user", content="I can start in two weeks"),
            Turn(
                role="assistant",
                content="Noted.",
                provider_transcript="I can start in two weeks.",
            ),
        ],
        ["yes"],
    )

    assert metric.score == 1.0


def test_strict_mode_clamps_anything_short_of_perfect_to_zero():
    turns = []
    for index in range(2):
        turns.append(Turn(role="user", content=f"question {index}"))
        turns.append(
            Turn(
                role="assistant",
                content=f"answer {index}",
                provider_transcript=f"question {index}",
            )
        )

    metric, _ = _measure(turns, ["yes", "no"], strict_mode=True)

    assert metric.score == 0
    assert metric.success is False


# ------------------------------------------------------------------- prompt


def test_the_judge_is_shown_what_was_spoken_and_what_was_heard():
    _, judge = _measure(
        [
            Turn(role="user", content="Siobhan Kavanagh"),
            Turn(
                role="assistant",
                content="Hello Siobhan!",
                provider_transcript="Siobhan Cavanaugh",
            ),
        ],
        ["no"],
        include_reason=False,
    )

    prompt = judge.prompts[0]
    assert "Siobhan Kavanagh" in prompt
    assert "Siobhan Cavanaugh" in prompt
    assert "Hello Siobhan!" in prompt


# -------------------------------------------------------- missing the field


def test_a_conversation_with_no_transcription_anywhere_is_refused():
    """A text conversation, or a phone call, can never satisfy this metric —
    saying so beats scoring it zero and looking like the agent failed."""
    metric = TranscriptionAccuracyMetric(
        model=ScriptedJudge(["yes"]), async_mode=False
    )

    with pytest.raises(MissingTestCaseParamsError) as excinfo:
        metric.measure(
            ConversationalTestCase(
                turns=[
                    Turn(role="user", content="hello"),
                    Turn(role="assistant", content="hi there"),
                ]
            )
        )

    assert "provider_transcript" in str(excinfo.value)


def test_one_transcribed_turn_is_enough_to_run():
    metric, _ = _measure(
        [
            Turn(role="user", content="first"),
            Turn(role="assistant", content="unmeasured"),
            Turn(role="user", content="second"),
            Turn(
                role="assistant",
                content="measured",
                provider_transcript="second",
            ),
        ],
        ["yes"],
    )

    assert metric.score == 1.0


# --------------------------------------------------------------- audio mode


class StubSTT(DeepEvalBaseSTT):
    """Returns one scripted transcription per clip and records the clips."""

    def __init__(self, heard, cost=None):
        self.scripted = list(heard)
        self.cost = cost
        self.clips = []
        super().__init__(model="stub-stt")

    def load_model(self):
        return self

    def transcribe(self, audio, **kwargs):
        self.clips.append(audio)
        return self.scripted.pop(0), self.cost

    async def a_transcribe(self, audio, **kwargs):
        return self.transcribe(audio, **kwargs)

    def get_model_name(self):
        return "stub-stt"


def _clip(start_time=None, seconds=0.02) -> Audio:
    pcm = b"\xe8\x03" * int(8000 * seconds)
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, sample_rate=8000),
        "audio/wav",
        sampleRate=8000,
        encoding="wav",
        duration=seconds,
        start_time=start_time,
    )


def _measure_audio(turns, scripted_verdicts, heard, **metric_kwargs):
    judge = ScriptedJudge(scripted_verdicts)
    stt = StubSTT(heard)
    metric = TranscriptionAccuracyMetric(
        model=judge,
        compare_against="audio",
        stt_model=stt,
        async_mode=metric_kwargs.pop("async_mode", False),
        **metric_kwargs,
    )
    metric.measure(ConversationalTestCase(turns=turns))
    return metric, judge, stt


def test_by_default_the_caller_content_is_compared_against():
    metric = TranscriptionAccuracyMetric(model=ScriptedJudge([]))

    assert metric.compare_against == "content"
    assert metric.stt_model is None
    assert (
        MultiTurnParams.PROVIDER_TRANSCRIPT in metric._required_test_case_params
    )


def test_an_stt_model_is_refused_outside_audio_mode():
    with pytest.raises(ValueError):
        TranscriptionAccuracyMetric(
            model=ScriptedJudge([]), stt_model=StubSTT([])
        )


def test_audio_mode_judges_both_the_caller_and_the_agent():
    pairs = get_audio_exchanges(
        [
            Turn(role="user", content="two weeks", audio=_clip(1.0)),
            Turn(
                role="assistant",
                content="Two weeks works.",
                audio=_clip(3.5),
            ),
        ]
    )

    assert [exchange for _, exchange in pairs] == [
        {
            "turn": 0,
            "role": "user",
            "spoken": None,
            "transcribed": "two weeks",
            "start_time": 1.0,
        },
        {
            "turn": 1,
            "role": "assistant",
            "spoken": None,
            "transcribed": "Two weeks works.",
            "start_time": 3.5,
        },
    ]


def test_a_caller_turn_uses_the_transcript_the_agent_reported():
    pairs = get_audio_exchanges(
        [
            Turn(role="user", content="two weeks", audio=_clip()),
            Turn(
                role="assistant",
                content="Two months works.",
                provider_transcript="two months",
            ),
        ]
    )

    assert [exchange["transcribed"] for _, exchange in pairs] == ["two months"]


def test_a_caller_turn_with_its_own_transcript_uses_it():
    pairs = get_audio_exchanges(
        [
            Turn(
                role="user",
                content="two weeks",
                provider_transcript="two years",
                audio=_clip(),
            ),
            Turn(
                role="assistant",
                content="Noted.",
                provider_transcript="two months",
            ),
        ]
    )

    assert [exchange["transcribed"] for _, exchange in pairs] == ["two years"]


def test_an_agent_turn_uses_what_it_meant_to_say_when_recorded():
    pairs = get_audio_exchanges(
        [
            Turn(
                role="assistant",
                content="Your interview is at",
                metadata={"intended_content": "Your interview is at 3."},
                audio=_clip(),
            ),
        ]
    )

    assert [exchange["transcribed"] for _, exchange in pairs] == [
        "Your interview is at 3."
    ]


def test_consecutive_caller_turns_fall_back_to_their_own_content():
    pairs = get_audio_exchanges(
        [
            Turn(role="user", content="first", audio=_clip()),
            Turn(role="user", content="second", audio=_clip()),
            Turn(
                role="assistant",
                content="Noted.",
                provider_transcript="first second",
            ),
        ]
    )

    assert [exchange["transcribed"] for _, exchange in pairs] == [
        "first",
        "second",
    ]


def test_turns_without_audio_are_skipped_in_audio_mode():
    pairs = get_audio_exchanges(
        [
            Turn(role="user", content="no clip"),
            Turn(role="assistant", content="clip", audio=_clip()),
            Turn(role="user", content="clip too", audio=_clip()),
        ]
    )

    assert [exchange["turn"] for _, exchange in pairs] == [1, 2]


def test_an_empty_transcript_with_audio_is_still_judged():
    pairs = get_audio_exchanges([Turn(role="user", content="", audio=_clip())])

    assert [exchange["transcribed"] for _, exchange in pairs] == [""]


def test_audio_mode_refuses_a_conversation_with_no_audio():
    with pytest.raises(MissingTestCaseParamsError) as excinfo:
        _measure_audio(
            [
                Turn(role="user", content="hi"),
                Turn(
                    role="assistant",
                    content="hello",
                    provider_transcript="hi",
                ),
            ],
            [],
            [],
        )

    assert "audio" in str(excinfo.value)


def test_audio_mode_does_not_need_a_provider_transcript():
    metric, _, _ = _measure_audio(
        [Turn(role="user", content="hi", audio=_clip())],
        ["yes"],
        ["hi"],
    )

    assert metric.score == 1.0


def test_the_score_is_the_share_of_turns_accounted_for():
    metric, _, stt = _measure_audio(
        [
            Turn(role="user", content="", audio=_clip()),
            Turn(role="assistant", content="Hello.", audio=_clip()),
            Turn(role="user", content="bye", audio=_clip()),
        ],
        ["no", "yes", "yes"],
        ["uh oh wait", "Hello.", "bye"],
    )

    assert metric.score == pytest.approx(2 / 3)
    assert len(stt.clips) == 3


def test_the_judge_is_shown_what_was_heard_and_when():
    _, judge, _ = _measure_audio(
        [Turn(role="user", content="", audio=_clip(start_time=42.3))],
        ["no"],
        ["uh oh wait"],
    )

    prompt = judge.prompts[0]
    assert "uh oh wait" in prompt
    assert "42.3" in prompt
    assert "'user'" in prompt
    assert "recorded voice call" in prompt
    assert "how accurately a voice agent's speech-to-text heard" not in prompt


def test_stt_cost_is_accrued_only_when_reported():
    turns = [
        Turn(role="user", content="a", audio=_clip()),
        Turn(role="assistant", content="b", audio=_clip()),
    ]

    metric = TranscriptionAccuracyMetric(
        model=ScriptedJudge([]),
        compare_against="audio",
        stt_model=StubSTT(["a", "b"], cost=0.01),
    )
    metric.evaluation_cost = 0.0
    metric._transcribe_exchanges(get_audio_exchanges(turns))
    assert metric.evaluation_cost == pytest.approx(0.02)

    metric = TranscriptionAccuracyMetric(
        model=ScriptedJudge([]),
        compare_against="audio",
        stt_model=StubSTT(["a", "b"]),
    )
    metric.evaluation_cost = 0.0
    metric._transcribe_exchanges(get_audio_exchanges(turns))
    assert metric.evaluation_cost == 0.0


def test_async_audio_mode_transcribes_every_clip():
    _, _, stt = _measure_audio(
        [
            Turn(role="user", content="a", audio=_clip()),
            Turn(role="assistant", content="b", audio=_clip()),
        ],
        ["yes", "yes"],
        ["a", "b"],
        async_mode=True,
    )

    assert len(stt.clips) == 2


def test_a_hybrid_audio_verdict_names_who_spoke_and_when():
    verdict = _verdict_for_audio_exchange(
        {
            "turn": 0,
            "role": "user",
            "spoken": "uh oh wait",
            "transcribed": "",
            "start_time": 42.3,
        },
        Verdict.NO,
        0.2,
    )

    assert "caller" in verdict.reason
    assert "at 42.3s" in verdict.reason
    assert "'uh oh wait'" in verdict.reason


def test_the_stt_model_is_named_in_the_verbose_logs():
    metric, _, _ = _measure_audio(
        [Turn(role="user", content="hi", audio=_clip())],
        ["yes"],
        ["hi"],
        verbose_mode=True,
    )

    assert "stub-stt" in metric.verbose_logs


def test_the_default_openai_stt_is_asked_to_transcribe_verbatim(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    metric = TranscriptionAccuracyMetric(
        model=ScriptedJudge([]), compare_against="audio"
    )

    assert isinstance(metric.stt_model, OpenAISTTModel)
    assert (
        metric.stt_model.transcription_kwargs["prompt"]
        == VERBATIM_TRANSCRIPTION_PROMPT
    )


def test_an_stt_model_passed_in_is_left_as_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stt = OpenAISTTModel()

    metric = TranscriptionAccuracyMetric(
        model=ScriptedJudge([]), compare_against="audio", stt_model=stt
    )

    assert metric.stt_model is stt
    assert "prompt" not in stt.transcription_kwargs


def test_whisper_is_not_asked_to_transcribe_verbatim(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    metric = TranscriptionAccuracyMetric(
        model=ScriptedJudge([]), compare_against="audio", stt_model="whisper-1"
    )

    assert "prompt" not in metric.stt_model.transcription_kwargs
