"""Issue #3654: QAG verdict coercion silently drops recoverable judge replies.

Fail-before: on current main, single-letter shorthands ("Y"/"N") and
uncertain words ("maybe"/"unsure"/"unknown"/"partial") all coerce to None,
so a judge answering in those words produces an empty verdict list and
`score_qag_verdicts` returns the empty-branch score (1.0 by default) — the
parse failure is reported as success.
"""

from typing import List, Optional

from pydantic import BaseModel

from deepeval.config.eval_mode import resolve_eval_mode
from deepeval.metrics.base_metric import (
    YES_NO,
    YES_NO_BORDERLINE,
    BaseMetric,
    Verdict,
)
from deepeval.metrics.utils import (
    a_generate_qag_verdict,
    a_generate_qag_verdicts,
    generate_qag_verdict,
    generate_qag_verdicts,
    initialize_model,
    initialize_system_one_model,
)
from deepeval.metrics.utils.qag import (
    _verdicts_from_json,
    normalize_qag_verdict,
    score_qag_verdicts,
)
from deepeval.metrics.utils.test_case import prepare_measure
from deepeval.metrics.utils.verbose import construct_verbose_logs
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.test_metrics.system_one_fakes import CannedLLM


class QAGVerdict(BaseModel):
    verdict: Verdict
    reason: Optional[str] = None


class QAGVerdicts(BaseModel):
    verdicts: List[QAGVerdict]


class _QAGStubMetric(BaseMetric):
    """Minimal QAG metric: a fake judge answering raw JSON, so the
    loose-JSON parsing (and its drop diagnostics) can be exercised with no
    network and no real model."""

    _required_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

    def __init__(self, model):
        self.eval_mode = resolve_eval_mode(None)
        self.model, self.using_native_model = initialize_model(
            model, self.eval_mode
        )
        self.system_one_model = initialize_system_one_model(
            None, self.eval_mode
        )
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = False
        self.strict_mode = False
        self.threshold = 0.5
        self.async_mode = False
        self.verbose_mode = False

    def measure(self, test_case, *args, **kwargs):
        raise NotImplementedError

    async def a_measure(self, test_case, *args, **kwargs):
        raise NotImplementedError

    @property
    def __name__(self):
        return "QAGStub"


class _StubMetric:
    strict_mode = False
    threshold = 0.5


def test_single_letter_shorthands_parse():
    assert normalize_qag_verdict("Y") == Verdict.YES
    assert normalize_qag_verdict("N") == Verdict.NO
    assert normalize_qag_verdict("y") == Verdict.YES
    assert normalize_qag_verdict("n") == Verdict.NO
    assert normalize_qag_verdict("Y, the answer is relevant") == Verdict.YES


def test_uncertain_words_map_to_borderline():
    for word in ("maybe", "unsure", "unknown", "partial", "idk"):
        assert normalize_qag_verdict(word, YES_NO_BORDERLINE) == (
            Verdict.BORDERLINE
        ), word
    # Borderline is not in the strict YES_NO vocabulary, so it still drops
    # there — the vocabulary, not the coercion, decides.
    for word in ("maybe", "unsure", "unknown", "partial", "idk"):
        assert normalize_qag_verdict(word, YES_NO) is None, word


def test_non_latin_verdicts_stay_dropped_by_design():
    # No silent acceptance of non-Latin verdicts: the issue asks for a
    # documented opt-in mapping, not a guess. They stay None (but must be
    # counted — see test_dropped_verdicts_are_counted).
    assert normalize_qag_verdict("是") is None
    assert normalize_qag_verdict("はい") is None
    assert normalize_qag_verdict("是", YES_NO_BORDERLINE) is None


def test_existing_coercion_kept():
    assert normalize_qag_verdict("yes") == Verdict.YES
    assert normalize_qag_verdict("no") == Verdict.NO
    assert normalize_qag_verdict("No, it's off-topic") == Verdict.NO
    assert normalize_qag_verdict("idk", YES_NO_BORDERLINE) == Verdict.BORDERLINE
    assert normalize_qag_verdict("banana") is None
    assert normalize_qag_verdict("") is None
    assert normalize_qag_verdict(None) is None
    assert normalize_qag_verdict(Verdict.YES) == Verdict.YES
    assert normalize_qag_verdict(Verdict.BORDERLINE) is None
    assert (
        normalize_qag_verdict(Verdict.BORDERLINE, YES_NO_BORDERLINE)
        == Verdict.BORDERLINE
    )


def test_verdicts_from_json_keeps_y_n():
    verdicts = _verdicts_from_json(
        {
            "verdicts": [
                {"verdict": "Y", "reason": "relevant"},
                {"verdict": "N", "reason": "off-topic"},
            ]
        },
        QAGVerdict,
        YES_NO,
    )
    assert [v.verdict for v in verdicts] == [Verdict.YES, Verdict.NO]


def test_all_y_judge_scores_from_parsed_verdicts():
    verdicts = _verdicts_from_json(
        {
            "verdicts": [
                {"verdict": "Y", "reason": "r1"},
                {"verdict": "Y", "reason": "r2"},
            ]
        },
        QAGVerdict,
        YES_NO,
    )
    # The verdicts must actually be there — today this list is empty and
    # the 1.0 below comes from the empty branch instead.
    assert len(verdicts) == 2
    assert (
        score_qag_verdicts(_StubMetric(), verdicts, passing=(Verdict.YES,))
        == 1.0
    )


def test_maybe_judge_no_longer_scores_perfect():
    verdicts = _verdicts_from_json(
        {
            "verdicts": [
                {"verdict": "maybe", "reason": "r1"},
                {"verdict": "maybe", "reason": "r2"},
            ]
        },
        QAGVerdict,
        YES_NO_BORDERLINE,
    )
    assert len(verdicts) == 2
    # Borderline verdicts do not pass a YES-only metric: 0.0, not the
    # empty-branch 1.0 that today's silent drop produces.
    assert (
        score_qag_verdicts(_StubMetric(), verdicts, passing=(Verdict.YES,))
        == 0.0
    )


def test_single_letter_token_boundary():
    # "n/a" and "y-axis" are longer tokens, not shorthands — their first
    # letter must not become NO/YES.
    assert normalize_qag_verdict("n/a") is None
    assert normalize_qag_verdict("N/A") is None
    assert normalize_qag_verdict("y-axis") is None
    assert normalize_qag_verdict("n2") is None
    # A sentence-terminated shorthand still counts.
    assert normalize_qag_verdict("N.") == Verdict.NO
    assert normalize_qag_verdict("Y") == Verdict.YES


def test_drop_counts_recorded_on_metric():
    judge = CannedLLM(
        '{"verdicts": [{"verdict": "Y", "reason": "r1"}, '
        '{"verdict": "maybe", "reason": "r2"}, '
        '{"verdict": "\\u662f", "reason": "r3"}]}'
    )
    metric = _QAGStubMetric(judge)
    verdicts = generate_qag_verdicts(
        metric=metric,
        prompt="judge",
        verdict_cls=QAGVerdict,
        verdicts_cls=QAGVerdicts,
        allowed=YES_NO,
    )
    assert [v.verdict for v in verdicts] == [Verdict.YES]
    assert metric.qag_verdicts_received_count == 3
    # "maybe" is recognized as BORDERLINE but outside the YES_NO
    # vocabulary, so it counts as dropped; the non-Latin reply too.
    assert metric.qag_verdicts_dropped_count == 2


def test_counts_accumulate_across_batches():
    judge = CannedLLM(
        '{"verdicts": [{"verdict": "Y", "reason": "r"}, '
        '{"verdict": "maybe", "reason": "r"}]}'
    )
    metric = _QAGStubMetric(judge)
    for _ in range(2):
        generate_qag_verdicts(
            metric=metric,
            prompt="judge",
            verdict_cls=QAGVerdict,
            verdicts_cls=QAGVerdicts,
            allowed=YES_NO,
        )
    assert metric.qag_verdicts_received_count == 4
    assert metric.qag_verdicts_dropped_count == 2


def test_single_verdict_path_counts_drops():
    metric = _QAGStubMetric(CannedLLM('{"verdict": "maybe", "reason": "r"}'))
    verdict = generate_qag_verdict(
        metric=metric, prompt="judge", verdict_cls=QAGVerdict, allowed=YES_NO
    )
    assert verdict is None
    assert metric.qag_verdicts_received_count == 1
    assert metric.qag_verdicts_dropped_count == 1

    metric = _QAGStubMetric(CannedLLM('{"verdict": "Y", "reason": "r"}'))
    verdict = generate_qag_verdict(
        metric=metric, prompt="judge", verdict_cls=QAGVerdict, allowed=YES_NO
    )
    assert verdict is not None and verdict.verdict == Verdict.YES
    assert metric.qag_verdicts_received_count == 1
    assert metric.qag_verdicts_dropped_count == 0


async def test_async_list_path_counts_drops():
    judge = CannedLLM(
        '{"verdicts": [{"verdict": "N", "reason": "r1"}, '
        '{"verdict": "maybe", "reason": "r2"}]}'
    )
    metric = _QAGStubMetric(judge)
    verdicts = await a_generate_qag_verdicts(
        metric=metric,
        prompt="judge",
        verdict_cls=QAGVerdict,
        verdicts_cls=QAGVerdicts,
        allowed=YES_NO,
    )
    assert [v.verdict for v in verdicts] == [Verdict.NO]
    assert metric.qag_verdicts_received_count == 2
    assert metric.qag_verdicts_dropped_count == 1


async def test_async_single_verdict_path_counts_drops():
    metric = _QAGStubMetric(CannedLLM('{"verdict": "maybe", "reason": "r"}'))
    verdict = await a_generate_qag_verdict(
        metric=metric, prompt="judge", verdict_cls=QAGVerdict, allowed=YES_NO
    )
    assert verdict is None
    assert metric.qag_verdicts_received_count == 1
    assert metric.qag_verdicts_dropped_count == 1


def test_counters_reset_by_prepare_measure():
    metric = _QAGStubMetric(CannedLLM("{}"))
    metric.qag_verdicts_received_count = 5
    metric.qag_verdicts_dropped_count = 2
    prepare_measure(
        metric,
        LLMTestCase(input="What is X?", actual_output="An answer."),
    )
    assert metric.qag_verdicts_received_count == 0
    assert metric.qag_verdicts_dropped_count == 0


def test_verbose_logs_show_drop_diagnostic():
    metric = _QAGStubMetric(CannedLLM("{}"))
    metric.qag_verdicts_received_count = 20
    metric.qag_verdicts_dropped_count = 3
    logs = construct_verbose_logs(
        metric, ["Verdicts:\n[]", "Score: 1.0\nReason: none"]
    )
    assert "3 of 20" in logs
    assert "unrecognized or unsupported" in logs


def test_verbose_logs_omit_diagnostic_without_drops():
    metric = _QAGStubMetric(CannedLLM("{}"))
    logs = construct_verbose_logs(
        metric, ["Verdicts:\n[]", "Score: 1.0\nReason: none"]
    )
    assert "QAG verdict parsing" not in logs
