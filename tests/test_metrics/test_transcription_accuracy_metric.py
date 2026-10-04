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
from deepeval.metrics.transcription_accuracy.transcription_accuracy import (
    get_transcribed_exchanges,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


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
                provider_transcription="I can start in two months",
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
                provider_transcription="hold on i meant two weeks",
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
                provider_transcription="second question",
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
                provider_transcription="",
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
            Turn(role="assistant", content="Sorry?", provider_transcription=""),
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
                provider_transcription=f"question {index}",
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
                provider_transcription="I can start in two weeks.",
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
                provider_transcription=f"question {index}",
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
                provider_transcription="Siobhan Cavanaugh",
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

    assert "provider_transcription" in str(excinfo.value)


def test_one_transcribed_turn_is_enough_to_run():
    metric, _ = _measure(
        [
            Turn(role="user", content="first"),
            Turn(role="assistant", content="unmeasured"),
            Turn(role="user", content="second"),
            Turn(
                role="assistant",
                content="measured",
                provider_transcription="second",
            ),
        ],
        ["yes"],
    )

    assert metric.score == 1.0
