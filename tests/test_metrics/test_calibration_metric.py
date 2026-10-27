"""Tests for CalibrationMetric and the pure helpers behind it.

No API keys, network or LLM: everything here is plain statistics.

Fixture: 16 anonymised (distance, outcome) pairs from a labelled retrieval
run. Confidence = f(distance) for an ARBITRARY monotone mapping f (two are
used below on purpose: ECE moves from 0.21 to 0.32 between them). Outcome = 1
only when the system returned the ground-truth answer; a query with no
ground truth counts as 0. The fixture validates the metric's mechanics, not
any claim about the source system's calibration quality.
"""

import asyncio
import random
import threading
from fractions import Fraction

import pytest

from deepeval import assert_test, evaluate
from deepeval.errors import MissingTestCaseParamsError
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics.community.calibration.calibration import (
    CalibrationMetric,
)
from deepeval.metrics.community.calibration.calibration_math import (
    MAX_N_BINS,
    calculate_calibration,
    calibration_bins,
)
from deepeval.metrics.community.calibration.schema import (
    CalibrationBin,
    CalibrationResult,
)
from deepeval.metrics.community.calibration.store import (
    create_calibration_store,
)
from deepeval.metrics.utils.metrics import copy_metrics
from deepeval.test_case import LLMTestCase

DISTANCES = [
    0.917121,
    0.653099,
    0.655621,
    0.389800,
    0.307194,
    0.447059,
    0.954232,
    0.604625,
    0.670573,
    1.887151,
    1.924156,
    1.909648,
    1.857025,
    0.930537,
    1.250613,
    0.964541,
]
OUTCOMES = [1] * 9 + [0] * 4 + [1, 0, 0]


def _linear(d):
    return max(0.0, min(1.0, 1 - d / 2))


def _inverse(d):
    return 1 / (1 + d)


MAPPINGS = [
    pytest.param(_linear, 0.2083, 0.1003, id="linear"),
    pytest.param(_inverse, 0.3225, 0.1549, id="inverse"),
]
ROWS = [(_linear(d), o) for d, o in zip(DISTANCES, OUTCOMES)]
EXPECTED_ECE = 0.2083
EXPECTED_BRIER = 0.1003
# Brier of always predicting the base rate (10/16): p * (1 - p).
BASELINE_BRIER = 0.2344


def make_case(conf, correct, name=None):
    return LLMTestCase(
        input="q",
        actual_output="a",
        name=name,
        metadata={"confidence": conf, "correct": correct},
    )


def named_cases(rows=ROWS):
    return [make_case(c, o, name=f"case-{i}") for i, (c, o) in enumerate(rows)]


def run_sync(metric, cases):
    for tc in cases:
        metric.measure(tc, _show_indicator=False)


def run_evaluate(metric, cases, run_async=True, **error_flags):
    return evaluate(
        test_cases=cases,
        metrics=[metric],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(**error_flags),
    )


# =============================================================================
# Pure math core
# =============================================================================


def test_perfect_confidence_gives_zero_error():
    assert calculate_calibration([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == (
        0.0,
        0.0,
    )


def test_confidently_wrong_gives_maximum_error():
    assert calculate_calibration([1.0, 0.0], [0, 1]) == (1.0, 1.0)


def test_constant_half_confidence_hand_computed():
    ece, brier = calculate_calibration([0.5] * 4, [1, 0, 1, 0])
    assert ece == pytest.approx(0.0)
    assert brier == pytest.approx(0.25)


def test_two_bin_hand_computed():
    # bin 1: n=2, acc 0.0, conf 0.1 -> 0.5 * 0.1
    # bin 9: n=2, acc 0.5, conf 0.9 -> 0.5 * 0.4
    ece, brier = calculate_calibration([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 0])
    assert ece == pytest.approx(0.25)
    assert brier == pytest.approx((0.01 + 0.01 + 0.01 + 0.81) / 4)


def test_single_bin_ece_is_gap_between_accuracy_and_mean_confidence():
    ece, brier = calculate_calibration([0.2, 0.6, 1.0], [0, 1, 1], n_bins=1)
    assert ece == pytest.approx(2 / 3 - 0.6)
    assert brier == pytest.approx((0.04 + 0.16 + 0.0) / 3)


def test_bool_and_int_outcomes_and_int_confidences_are_equivalent():
    assert calculate_calibration([0.3, 0.8], [True, False]) == (
        calculate_calibration([0.3, 0.8], [1, 0])
    )
    assert calculate_calibration([0, 1], [0, 1]) == (0.0, 0.0)


def test_bin_stats_structure():
    bins = calibration_bins([0.05, 0.15, 0.15], [1, 0, 1], n_bins=10)
    assert len(bins) == 10
    assert sum(b.count for b in bins) == 3
    assert (bins[0].count, bins[0].avg_confidence, bins[0].accuracy) == (
        1,
        0.05,
        1.0,
    )
    assert bins[1].count == 2
    assert bins[1].avg_confidence == pytest.approx(0.15)
    assert bins[1].accuracy == pytest.approx(0.5)
    assert (bins[5].count, bins[5].avg_confidence, bins[5].accuracy) == (
        0,
        0.0,
        0.0,
    )
    assert (bins[3].lower, bins[3].upper) == (0.3, 0.4)


@pytest.mark.parametrize("n", range(1, 41))
def test_values_on_bin_edges_land_in_the_bin_they_open(n):
    for k in range(n + 1):
        c = k / n
        bins = calibration_bins([c], [1], n)
        hit = [i for i, b in enumerate(bins) if b.count == 1]
        assert hit == [min(k, n - 1)], (n, k)
        assert bins[hit[0]].lower <= c <= bins[hit[0]].upper


def test_float_error_edge_case_31_over_39():
    # Bare int(c * n) would put this value in bin 30 (31/39*39 < 31).
    assert (31 / 39) * 39 < 31
    assert calibration_bins([31 / 39], [1], 39)[31].count == 1


def _exact_reference(confs, outs, n):
    """Independent implementation using exact rational arithmetic."""
    total = len(confs)
    groups = [[] for _ in range(n)]
    for c, o in zip(confs, outs):
        groups[min(int(Fraction(c) * n), n - 1)].append((Fraction(c), o))
    ece = Fraction(0)
    for g in groups:
        if g:
            acc = Fraction(sum(o for _, o in g), len(g))
            conf = sum(c for c, _ in g) / len(g)
            ece += Fraction(len(g), total) * abs(acc - conf)
    brier = sum((Fraction(c) - o) ** 2 for c, o in zip(confs, outs)) / total
    return float(ece), float(brier)


@pytest.mark.parametrize("seed", range(20))
def test_matches_exact_reference_and_invariants_on_random_data(seed):
    rng = random.Random(seed)
    rows = [
        (rng.random(), rng.randint(0, 1)) for _ in range(rng.randint(1, 60))
    ]
    n = rng.randint(1, 25)
    confs = [c for c, _ in rows]
    outs = [o for _, o in rows]

    ece, brier = calculate_calibration(confs, outs, n)
    ref_ece, ref_brier = _exact_reference(confs, outs, n)
    assert ece == pytest.approx(ref_ece, abs=1e-12)
    assert brier == pytest.approx(ref_brier, abs=1e-12)
    assert 0.0 <= ece <= 1.0 and 0.0 <= brier <= 1.0
    assert sum(b.count for b in calibration_bins(confs, outs, n)) == len(rows)

    shuffled = rows[:]
    rng.shuffle(shuffled)
    assert calculate_calibration(
        [c for c, _ in shuffled], [o for _, o in shuffled], n
    ) == pytest.approx((ece, brier), abs=1e-12)


@pytest.mark.parametrize(
    "confs,outs,n_bins",
    [
        ([], [], 10),
        ([0.5], [1, 0], 10),
        ([float("nan")], [1], 10),
        ([float("inf")], [1], 10),
        ([1.5], [1], 10),
        ([-0.1], [1], 10),
        ([10**1000], [1], 10),  # float() would overflow: must stay ValueError
        ([True], [1], 10),
        (["0.5"], [1], 10),
        ([None], [1], 10),
        ([0.5], [2], 10),
        ([0.5], [0.5], 10),
        ([0.5], [None], 10),
        ([0.5], [1], 0),
        ([0.5], [1], -3),
        ([0.5], [1], True),
        ([0.5], [1], 2.5),
        ([0.5], [1], MAX_N_BINS + 1),
        ([0.5], [1], 10**7),
    ],
)
def test_math_core_rejects_invalid_input(confs, outs, n_bins):
    with pytest.raises(ValueError):
        calculate_calibration(confs, outs, n_bins)


def test_math_core_accepts_max_n_bins():
    bins = calibration_bins([0.5], [1], MAX_N_BINS)
    assert len(bins) == MAX_N_BINS


# =============================================================================
# Run store
# =============================================================================


def test_store_add_returns_row_count_and_none_key_appends():
    store = create_calibration_store()
    assert store.add(0.5, 1) == 1
    assert store.add(0.5, 1) == 2
    assert len(store) == 2


def test_store_same_key_overwrites_and_keeps_position():
    store = create_calibration_store()
    store.add(0.1, 0, key="a")
    store.add(0.2, 1, key="b")
    store.add(0.3, 0, key="a")
    assert len(store) == 2
    assert store.snapshot() == ([0.3, 0.2], [0, 1])


def test_store_snapshot_is_a_copy():
    store = create_calibration_store()
    store.add(0.5, 1)
    confs, outs = store.snapshot()
    confs.append(0.9)
    outs.append(0)
    assert store.snapshot() == ([0.5], [1])


@pytest.mark.parametrize(
    "conf,outcome",
    [(1.5, 1), (float("nan"), 1), (True, 1), ("0.5", 1), (0.5, 2), (0.5, None)],
)
def test_store_rejects_bad_rows_and_stays_unchanged(conf, outcome):
    store = create_calibration_store()
    store.add(0.5, 1)
    with pytest.raises(ValueError):
        store.add(conf, outcome)
    assert len(store) == 1


def test_store_reset_and_factory_independence():
    a, b = create_calibration_store(), create_calibration_store()
    a.add(0.5, 1)
    assert len(b) == 0
    a.reset()
    assert len(a) == 0


def test_store_empty_aggregate_propagates_math_error():
    with pytest.raises(ValueError):
        create_calibration_store().aggregate(calculate_calibration)


def test_store_concurrent_adds_lose_no_rows():
    store = create_calibration_store()

    def worker(offset):
        for i in range(500):
            store.add(0.5, i % 2, key=(offset, i))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 4000


def test_store_aggregate_runs_outside_the_lock():
    store = create_calibration_store()
    store.add(0.5, 1)
    seen = []

    # len(store) takes the (non-reentrant) lock: would deadlock if
    # aggregate() still held it while running fn.
    t = threading.Thread(
        target=lambda: seen.append(store.aggregate(lambda c, o: len(store))),
        daemon=True,
    )
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert seen == [1]


# =============================================================================
# Schema
# =============================================================================


def test_result_is_frozen():
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    res = m.get_calibration_result()
    with pytest.raises(ValueError):
        res.ece = 0.0


def test_result_requires_outcome_definition():
    with pytest.raises(ValueError):
        CalibrationResult(ece=0.0, brier_score=0.0, n_bins=1, n_samples=1)


def test_result_validates_bins():
    good = CalibrationBin(
        lower=0.0, upper=0.5, count=1, avg_confidence=0.2, accuracy=1.0
    )
    base = dict(ece=0.0, brier_score=0.0, n_samples=1, outcome_definition="x")
    with pytest.raises(ValueError):  # wrong number of bins
        CalibrationResult(n_bins=2, bins=[good], **base)
    with pytest.raises(ValueError):  # counts do not sum to n_samples
        CalibrationResult(n_bins=1, **{**base, "n_samples": 2}, bins=[good])
    with pytest.raises(ValueError):  # empty bin with non-zero stats
        CalibrationBin(
            lower=0.0, upper=0.5, count=0, avg_confidence=0.2, accuracy=0.0
        )


# =============================================================================
# Metric: aggregate correctness
# =============================================================================


@pytest.mark.parametrize("mapping,ece,brier", MAPPINGS)
def test_aggregate_matches_reference_values(mapping, ece, brier):
    rows = [(mapping(d), o) for d, o in zip(DISTANCES, OUTCOMES)]
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases(rows))
    res = m.get_calibration_result()
    assert res.n_samples == 16
    assert res.ece == pytest.approx(ece, abs=5e-4)
    assert res.brier_score == pytest.approx(brier, abs=5e-4)
    assert res.brier_score < BASELINE_BRIER
    assert len(res.bins) == res.n_bins == 10
    assert "answer was correct" in res.outcome_definition


def test_metric_equals_standalone_math_exactly():
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    res = m.get_calibration_result()
    ece, brier = calculate_calibration(
        [c for c, _ in ROWS], [o for _, o in ROWS]
    )
    assert (res.ece, res.brier_score) == (ece, brier)


def test_aggregate_independent_of_arrival_order():
    rows = list(ROWS)
    random.Random(0).shuffle(rows)
    m = CalibrationMetric(async_mode=False)
    run_sync(m, [make_case(c, o) for c, o in rows])
    res = m.get_calibration_result()
    assert res.ece == pytest.approx(EXPECTED_ECE, abs=5e-4)
    assert res.brier_score == pytest.approx(EXPECTED_BRIER, abs=5e-4)


def test_rows_accumulate_per_measure_call():
    m = CalibrationMetric(async_mode=False)
    for i, tc in enumerate(named_cases(), start=1):
        m.measure(tc, _show_indicator=False)
        assert m.score_breakdown["n_samples"] == i


def test_custom_n_bins():
    m = CalibrationMetric(async_mode=False, n_bins=5)
    run_sync(m, named_cases())
    res = m.get_calibration_result()
    assert res.n_bins == len(res.bins) == 5


def test_reason_summarises_the_aggregate():
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    assert "ECE=0.2083" in m.reason
    assert "Brier=0.1003" in m.reason
    assert "over 16 rows" in m.reason


# =============================================================================
# Metric: per-case semantics
# =============================================================================


def test_per_case_score_is_one_minus_squared_error():
    m = CalibrationMetric(async_mode=False)
    assert m.measure(make_case(0.5, 1), _show_indicator=False) == (
        pytest.approx(0.75)
    )
    assert m.measure(make_case(0.9, 0), _show_indicator=False) == (
        pytest.approx(0.19)
    )
    assert m.success is False  # 0.19 < default threshold 0.5


def test_custom_threshold_changes_success():
    m = CalibrationMetric(async_mode=False, threshold=0.1)
    m.measure(make_case(0.9, 0), _show_indicator=False)
    assert m.success is True


def test_strict_mode_zeroes_scores_below_threshold():
    m = CalibrationMetric(async_mode=False, strict_mode=True)
    assert m.measure(make_case(0.5, 1), _show_indicator=False) == 0.0
    assert m.measure(make_case(1.0, 1), _show_indicator=False) == 1.0


def test_include_reason_false_gives_no_reason():
    m = CalibrationMetric(async_mode=False, include_reason=False)
    m.measure(make_case(0.5, 1), _show_indicator=False)
    assert m.reason is None


def test_custom_metadata_keys():
    m = CalibrationMetric(
        async_mode=False, confidence_key="p", correct_key="ok"
    )
    tc = LLMTestCase(
        input="q", actual_output="a", metadata={"p": 0.8, "ok": True}
    )
    assert m.measure(tc, _show_indicator=False) == pytest.approx(0.96)


def test_numpy_scalars_are_accepted():
    np = pytest.importorskip("numpy")
    m = CalibrationMetric(async_mode=False)
    m.measure(make_case(np.float32(0.5), np.bool_(True)), _show_indicator=False)
    m.measure(make_case(np.float64(0.25), np.int64(0)), _show_indicator=False)
    assert m.get_calibration_result().n_samples == 2


# =============================================================================
# Metric: run store, identity, reruns, isolation
# =============================================================================


def test_named_cases_make_reruns_idempotent():
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    run_sync(m, named_cases())
    assert m.get_calibration_result().n_samples == 16


def test_unnamed_cases_append_on_rerun():
    m = CalibrationMetric(async_mode=False)
    run_sync(m, [make_case(c, o) for c, o in ROWS])
    run_sync(m, [make_case(c, o) for c, o in ROWS])
    assert m.get_calibration_result().n_samples == 32


def test_reset_starts_a_new_run():
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    m.reset()
    run_sync(m, named_cases()[:3])
    assert m.get_calibration_result().n_samples == 3


@pytest.mark.parametrize("mode", ["measure", "sync_evaluate", "async_evaluate"])
def test_same_name_different_outcome_keeps_both_rows(mode):
    # Scenario from review feedback on #3485: distinct cases sharing a name.
    m = CalibrationMetric(async_mode=False)
    cases = [
        make_case(0.9, 1, name="example"),
        make_case(0.9, 0, name="example"),
    ]
    if mode == "measure":
        run_sync(m, cases)
    else:
        run_evaluate(m, cases, run_async=(mode == "async_evaluate"))
    result = m.get_calibration_result()
    assert result.n_samples == 2
    assert result.brier_score == pytest.approx(0.41)  # (0.01 + 0.81) / 2
    assert result.ece == pytest.approx(0.4)  # |0.9 - 0.5|


def test_same_name_same_numbers_different_input_keeps_both_rows():
    m = CalibrationMetric(async_mode=False)
    cases = [
        LLMTestCase(
            input=q,
            actual_output="a",
            name="example",
            metadata={"confidence": 0.9, "correct": 1},
        )
        for q in ("q1", "q2")
    ]
    run_sync(m, cases)
    result = m.get_calibration_result()
    assert result.n_samples == 2
    assert result.brier_score == pytest.approx(0.01)
    assert result.ece == pytest.approx(0.1)


def test_two_metrics_do_not_share_state():
    a, b = CalibrationMetric(async_mode=False), CalibrationMetric(
        async_mode=False
    )
    run_sync(a, named_cases())
    assert a.run_store is not b.run_store
    assert len(b.run_store) == 0


def test_injected_store_is_used():
    store = create_calibration_store()
    m = CalibrationMetric(run_store=store, async_mode=False)
    run_sync(m, named_cases())
    assert len(store) == 16


# =============================================================================
# Metric: DeepEval integration (copy_metrics / evaluate / assert_test)
# =============================================================================


def test_copy_metrics_shares_store_and_forwards_all_config():
    m = CalibrationMetric(
        threshold=0.6,
        confidence_key="p",
        correct_key="ok",
        n_bins=5,
        include_reason=False,
        async_mode=False,
        strict_mode=True,
        verbose_mode=True,
        flaky=True,
    )
    (copy,) = copy_metrics([m])
    assert copy is not m
    assert copy.run_store is m.run_store
    assert (
        copy.confidence_key,
        copy.correct_key,
        copy.n_bins,
        copy.include_reason,
        copy.async_mode,
        copy.strict_mode,
        copy.verbose_mode,
        copy.flaky,
    ) == ("p", "ok", 5, False, False, True, True, True)
    assert copy.threshold == 1  # strict_mode forces threshold to 1


def test_copy_metrics_drops_store_if_attr_name_mismatches():
    """Documents the silent-drop failure mode: the store must live under the
    same attribute name as the __init__ parameter (`run_store`), otherwise
    copy_metrics() gives each copy a fresh store (ECE over 1 row)."""
    m = CalibrationMetric()
    original = m.run_store
    m.renamed_store = m.run_store
    del m.run_store  # attr no longer matches the init param name
    (copy,) = copy_metrics([m])
    assert copy.run_store is not original


def test_async_evaluate_aggregates_all_rows_on_the_users_instance():
    m = CalibrationMetric()
    run_evaluate(m, named_cases(), run_async=True)
    res = m.get_calibration_result()
    assert res.n_samples == 16
    assert res.ece == pytest.approx(EXPECTED_ECE, abs=5e-4)
    assert res.brier_score == pytest.approx(EXPECTED_BRIER, abs=5e-4)


def test_sync_evaluate_reuses_the_users_instance():
    m = CalibrationMetric(async_mode=False)
    run_evaluate(m, named_cases(), run_async=False)
    assert m.score_breakdown["n_samples"] == 16
    assert m.get_calibration_result().n_samples == 16


def test_async_evaluate_copies_the_metric_per_case():
    """Regression for the shared-instance finding: in async mode the user's
    instance is never measured itself (DeepEval copies it per test case), so
    the aggregate must be read from the shared store, not from m.score."""
    m = CalibrationMetric()
    run_evaluate(m, named_cases(), run_async=True)
    assert m.score is None
    assert m.get_calibration_result().n_samples == 16


def test_concurrent_a_measure_loses_no_rows():
    m = CalibrationMetric()
    rows = [(i / 199, i % 2) for i in range(200)]

    async def main():
        await asyncio.gather(
            *[
                m.a_measure(
                    make_case(c, o, name=f"r{i}"), _show_indicator=False
                )
                for i, (c, o) in enumerate(rows)
            ]
        )

    asyncio.run(main())
    assert m.get_calibration_result().n_samples == 200


@pytest.mark.parametrize("run_async", [False, True])
def test_skip_on_missing_params_skips_the_row(run_async):
    m = CalibrationMetric(async_mode=run_async)
    cases = named_cases()[:3] + [
        LLMTestCase(
            input="q", actual_output="a", name="bad", metadata={"correct": 1}
        )
    ]
    run_evaluate(m, cases, run_async=run_async, skip_on_missing_params=True)
    assert m.get_calibration_result().n_samples == 3


@pytest.mark.parametrize("run_async", [False, True])
def test_ignore_errors_keeps_good_rows_and_drops_bad_ones(run_async):
    m = CalibrationMetric(async_mode=run_async)
    cases = named_cases()[:3] + [make_case(1.5, 1, name="bad")]
    run_evaluate(m, cases, run_async=run_async, ignore_errors=True)
    assert m.get_calibration_result().n_samples == 3


def test_missing_params_raise_through_evaluate_by_default():
    m = CalibrationMetric(async_mode=False)
    cases = [LLMTestCase(input="q", actual_output="a", name="bad")]
    with pytest.raises(MissingTestCaseParamsError):
        run_evaluate(m, cases, run_async=False)


def test_assert_test_passes_for_a_well_calibrated_case():
    assert_test(make_case(0.9, 1), [CalibrationMetric()])


def test_assert_test_fails_for_a_confident_wrong_case():
    with pytest.raises(AssertionError):
        assert_test(make_case(0.9, 0), [CalibrationMetric()])


# =============================================================================
# Metric: input validation
# =============================================================================


@pytest.mark.parametrize("meta", [None, {"confidence": 0.5}, {"correct": 1}])
def test_missing_metadata_raises_missing_params(meta):
    m = CalibrationMetric(async_mode=False)
    with pytest.raises(MissingTestCaseParamsError):
        m.measure(
            LLMTestCase(input="q", actual_output="a", metadata=meta),
            _show_indicator=False,
        )
    assert m.error
    assert len(m.run_store) == 0


@pytest.mark.parametrize(
    "conf,correct",
    [
        (1.5, 1),
        (-0.1, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        (10**1000, 1),  # float() overflows: must still be ValueError
        ("0.5", 1),
        (True, 1),
        (None, 1),
        (0.5, 2),
        (0.5, "yes"),
        (0.5, None),
    ],
)
def test_invalid_rows_raise_value_error_and_leave_store_unchanged(
    conf, correct
):
    m = CalibrationMetric(async_mode=False)
    with pytest.raises(ValueError):
        m.measure(make_case(conf, correct), _show_indicator=False)
    assert m.error
    assert len(m.run_store) == 0


@pytest.mark.parametrize("n_bins", [0, -1, 1001, 10**7, True, 2.5])
def test_invalid_n_bins_rejected(n_bins):
    with pytest.raises(ValueError):
        CalibrationMetric(n_bins=n_bins)


# =============================================================================
# Metric: warnings
# =============================================================================


def test_small_sample_warning_present_then_absent(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    assert any(
        w.startswith("Only 16 rows (< 30)")
        for w in m.get_calibration_result().warnings
    )
    run_sync(m, [make_case(0.5, i % 2) for i in range(20)])
    assert not m.get_calibration_result().warnings


def test_xdist_warning(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    assert any("xdist" in w for w in m.get_calibration_result().warnings)


def test_no_xdist_warning_without_the_env_var(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    m = CalibrationMetric(async_mode=False)
    run_sync(m, named_cases())
    assert not any("xdist" in w for w in m.get_calibration_result().warnings)
