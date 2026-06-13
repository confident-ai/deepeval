"""Deterministic tests for TurnRelevancyMetric scoring (issue #2321).

These tests use a scripted fake judge model and do NOT require any API key.
They pin down:
  - the score arithmetic: score = (# exchanges judged relevant) / (# exchanges judged)
  - one verdict per user->assistant unit interaction (not per raw turn)
  - trailing user turns (no assistant reply yet) are excluded from scoring
  - the verdict schema only admits 'yes'/'no'
  - a non-native judge returning a non-conforming verdict string is normalized
    (leading yes/no) or dropped, never crashing and never silently counted as
    relevant
  - the documented fallback when no valid verdicts exist (score = 1)
  - custom `evaluation_template` is honored
"""

import json

import pytest
from pydantic import ValidationError

from deepeval.metrics import TurnRelevancyMetric
from deepeval.metrics.turn_relevancy.schema import (
    TurnRelevancyVerdict,
    TurnRelevancyScoreReason,
)
from deepeval.metrics.turn_relevancy.template import TurnRelevancyTemplate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import ConversationalTestCase, Turn


class ScriptedJudge(DeepEvalBaseLLM):
    """Returns pre-scripted verdicts in order and records prompts."""

    def __init__(self, verdicts):
        self.scripted = list(verdicts)
        self.calls = 0
        self.prompts = []
        super().__init__(model="scripted-judge")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        self.prompts.append(prompt)
        if schema is TurnRelevancyScoreReason:
            return TurnRelevancyScoreReason(reason="scripted reason")
        verdict = self.scripted[self.calls % len(self.scripted)]
        self.calls += 1
        return TurnRelevancyVerdict(
            verdict=verdict,
            reason="scripted irrelevancy" if verdict == "no" else None,
        )

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "scripted-judge"


class RawJsonJudge(DeepEvalBaseLLM):
    """Non-native judge that returns raw JSON *strings*, exercising the
    string-extraction path that real custom/local models go through
    (as opposed to returning schema instances directly)."""

    def __init__(self, verdict_payloads):
        self.scripted = list(verdict_payloads)
        self.calls = 0
        super().__init__(model="raw-json-judge")

    def load_model(self):
        return self

    def generate(self, prompt, schema=None, **kwargs):
        if schema is TurnRelevancyScoreReason:
            return json.dumps({"reason": "scripted reason"})
        payload = self.scripted[self.calls % len(self.scripted)]
        self.calls += 1
        return payload

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return "raw-json-judge"


def _conversation(n_exchanges):
    turns = []
    for k in range(n_exchanges):
        turns.append(Turn(role="user", content=f"User question {k}"))
        turns.append(Turn(role="assistant", content=f"Assistant answer {k}"))
    return ConversationalTestCase(turns=turns)


def _measure(test_case, scripted_verdicts, **metric_kwargs):
    judge = ScriptedJudge(scripted_verdicts)
    metric = TurnRelevancyMetric(
        model=judge,
        async_mode=False,
        include_reason=False,
        **metric_kwargs,
    )
    score = metric.measure(
        test_case, _show_indicator=False, _log_metric_to_confident=False
    )
    return score, metric, judge


class TestTurnRelevancyAggregation:
    def test_single_irrelevant_exchange_scores_zero(self):
        test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content="What is X?"),
                Turn(
                    role="assistant",
                    content="This is unrelated and incorrect.",
                ),
            ]
        )
        score, metric, _ = _measure(test_case, ["no"])
        assert score == 0.0
        assert len(metric.verdicts) == 1

    @pytest.mark.parametrize(
        "n_exchanges, n_irrelevant, expected_score",
        [
            (5, 1, 0.8),
            (10, 1, 0.9),
            (10, 2, 0.8),
            (20, 1, 0.95),
            (4, 4, 0.0),
            (4, 0, 1.0),
        ],
    )
    def test_score_is_fraction_of_relevant_exchanges(
        self, n_exchanges, n_irrelevant, expected_score
    ):
        scripted = ["no"] * n_irrelevant + ["yes"] * (
            n_exchanges - n_irrelevant
        )
        score, metric, _ = _measure(_conversation(n_exchanges), scripted)
        assert score == pytest.approx(expected_score)
        assert len(metric.verdicts) == n_exchanges

    def test_one_verdict_per_unit_interaction_not_per_raw_turn(self):
        # 5 exchanges == 10 raw turns -> exactly 5 judge calls
        score, metric, judge = _measure(_conversation(5), ["yes"])
        assert judge.calls == 5

    def test_trailing_user_turn_is_not_scored(self):
        test_case = _conversation(2)
        test_case.turns.append(
            Turn(role="user", content="An unanswered follow-up?")
        )
        score, metric, _ = _measure(test_case, ["yes"])
        # only the 2 completed exchanges are judged
        assert len(metric.verdicts) == 2
        assert score == 1.0

    def test_verdict_schema_only_admits_yes_or_no(self):
        with pytest.raises(ValidationError):
            TurnRelevancyVerdict(verdict="No, irrelevant")
        with pytest.raises(ValidationError):
            TurnRelevancyVerdict(verdict="irrelevant")
        assert TurnRelevancyVerdict(verdict="no").verdict == "no"
        assert TurnRelevancyVerdict(verdict="yes").verdict == "yes"

    def test_no_valid_verdicts_falls_back_to_perfect_score(self):
        # Documents the fallback introduced for issue #2327: if every
        # verdict failed to generate, the metric scores 1 rather than crash.
        metric = TurnRelevancyMetric(
            model=ScriptedJudge(["yes"]), async_mode=False
        )
        metric.verdicts = [None, None, None]
        assert metric._calculate_score() == 1

    def test_failed_verdicts_are_excluded_from_denominator(self):
        metric = TurnRelevancyMetric(
            model=ScriptedJudge(["yes"]), async_mode=False
        )
        metric.verdicts = [
            None,
            TurnRelevancyVerdict(verdict="no", reason="irrelevant"),
            None,
            TurnRelevancyVerdict(verdict="yes"),
        ]
        assert metric._calculate_score() == 0.5

    def test_nonconforming_verdict_string_is_normalized_not_crashed(self):
        # A non-native judge that returns a raw JSON string with a verbose
        # verdict ("No, ...") goes through the string-extraction path. It must
        # not crash (Literal schema) and must be counted as irrelevant, not
        # silently passed as relevant (issue #2321).
        judge = RawJsonJudge(['{"verdict": "No, this is off-topic."}'])
        metric = TurnRelevancyMetric(
            model=judge, async_mode=False, include_reason=False
        )
        score = metric.measure(
            ConversationalTestCase(
                turns=[
                    Turn(role="user", content="What is X?"),
                    Turn(role="assistant", content="Completely unrelated."),
                ]
            ),
            _show_indicator=False,
            _log_metric_to_confident=False,
        )
        assert score == 0.0
        assert metric.verdicts[0] is not None
        assert metric.verdicts[0].verdict == "no"

    def test_ambiguous_verdict_string_is_dropped_not_crashed(self):
        # An unparseable verdict ("maybe") is excluded from scoring (None),
        # consistent with the #2327 fallback — and never raises.
        judge = RawJsonJudge(['{"verdict": "maybe"}'])
        metric = TurnRelevancyMetric(
            model=judge, async_mode=False, include_reason=False
        )
        score = metric.measure(
            ConversationalTestCase(
                turns=[
                    Turn(role="user", content="What is X?"),
                    Turn(role="assistant", content="An answer."),
                ]
            ),
            _show_indicator=False,
            _log_metric_to_confident=False,
        )
        assert metric.verdicts[0] is None
        assert score == 1  # no valid verdicts -> documented fallback

    def test_custom_evaluation_template_is_used(self):
        class StrictTemplate(TurnRelevancyTemplate):
            @staticmethod
            def generate_verdicts(sliding_window):
                return (
                    "CUSTOM_STRICT_RUBRIC\n"
                    + TurnRelevancyTemplate.generate_verdicts(sliding_window)
                )

        score, metric, judge = _measure(
            _conversation(1), ["yes"], evaluation_template=StrictTemplate
        )
        assert "CUSTOM_STRICT_RUBRIC" in judge.prompts[0]
