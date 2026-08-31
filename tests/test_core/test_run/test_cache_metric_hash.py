"""Tests for Cache identity fields (metric_class / metric_hash) and
source-code-based cache invalidation introduced by #3047.

Structure:
  * unit-level assertions on _compute_metric_hash determinism, the
    same_metric_configs equivalence relation, and backward compatibility
    with legacy cache entries (no hash present).
  * an integration-level round-trip through Cache.get_metric_data and
    Cache.create_metric_configuration that exercises the actual code
    paths used from evaluate(... use_cache=True).
"""

from __future__ import annotations

from typing import List

import pytest

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import MetricData
from deepeval.test_run.cache import (
    Cache,
    CachedMetricData,
    CachedTestCase,
    MetricConfiguration,
    _compute_metric_hash,
    _hashable_repr,
    _is_builtin_metric,
    _qualified_metric_class_name,
)


# ---------------------------------------------------------------------------
# Unit helpers for user-defined metrics. Each class lives inside its own
# module-compatible scope so inspect.getsource() can locate the body.
# ---------------------------------------------------------------------------


class _StrictnessZero(BaseMetric):
    """Baseline custom metric. `measure` returns a fixed score of 1."""

    name = "_StrictnessZero"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.success = False
        self.score = None
        self.strict_mode = False
        self.error = None

    def measure(self, test_case, *args, **kwargs):
        self.score = 1.0

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class _StrictnessZeroV2(BaseMetric):
    """Same class name prefix / same fields, different eval logic.

    If source-hashing is wired up, this should produce a DIFFERENT
    metric_hash than _StrictnessZero, despite sharing parameters.
    """

    name = "_StrictnessZero"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.success = False
        self.score = None
        self.strict_mode = False
        self.error = None

    def measure(self, test_case, *args, **kwargs):
        # Changed: score is now bounded by the input length.
        self.score = max(0.0, min(1.0, len(test_case.input or "") / 100.0))

    def is_successful(self) -> bool:
        return self.score >= self.threshold


class _CustomWithAsync(BaseMetric):
    """Metric with both sync and async eval methods. The async body also
    contributes to the digest — we verify that separately."""

    name = "_CustomWithAsync"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score = None
        self.error = None

    def measure(self, test_case, *args, **kwargs):
        self.score = 0.7

    async def a_measure(self, test_case, *args, **kwargs):
        self.score = 0.7

    def is_successful(self) -> bool:
        return self.score >= self.threshold


# ---------------------------------------------------------------------------
# _hashable_repr determinism
# ---------------------------------------------------------------------------


class TestHashableRepr:
    def test_dict_key_order_independent(self):
        a = {"a": 1, "b": "x", "c": None}
        b = {"b": "x", "c": None, "a": 1}
        assert _hashable_repr(a) == _hashable_repr(b)

    def test_none_distinct_from_empty_collection(self):
        # Guard against silent collapsing that would cause unrelated
        # configs to land on the same digest.
        assert _hashable_repr(None) != _hashable_repr([])
        assert _hashable_repr(None) != _hashable_repr({})
        assert _hashable_repr(0) != _hashable_repr(False)

    def test_nested_lists_roundtrip_stably(self):
        value = [None, [1, "foo"], {"k": (True, False)}]
        assert _hashable_repr(value) == _hashable_repr(
            [None, [1, "foo"], {"k": (True, False)}]
        )

    def test_bool_and_int_do_not_collide(self):
        # bool is a subclass of int — a naive branch order would collapse
        # them.
        assert _hashable_repr(True) != _hashable_repr(1)
        assert _hashable_repr(False) != _hashable_repr(0)


# ---------------------------------------------------------------------------
# _is_builtin_metric classifies correctly
# ---------------------------------------------------------------------------


class TestBuiltinClassification:
    def test_shipped_metric_is_builtin(self):
        from deepeval.metrics import ExactMatchMetric

        assert _is_builtin_metric(ExactMatchMetric)

    def test_user_defined_subclass_is_not_builtin(self):
        assert not _is_builtin_metric(_StrictnessZero)

    def test_dynamically_created_is_not_builtin(self):
        type_name = "_DynamicClass_"
        Dynamic = type(type_name, (BaseMetric,), {"__module__": "user_script"})
        assert not _is_builtin_metric(Dynamic)


# ---------------------------------------------------------------------------
# _compute_metric_hash contracts
# ---------------------------------------------------------------------------


class TestComputeMetricHash:
    def test_deterministic_for_same_config(self):
        a = _StrictnessZero(threshold=0.3)
        b = _StrictnessZero(threshold=0.3)
        fields = ("threshold", "evaluation_model", "strict_mode")
        payload_a = {f: getattr(a, f, None) for f in fields}
        payload_b = {f: getattr(b, f, None) for f in fields}
        assert _compute_metric_hash(a, payload_a) == _compute_metric_hash(
            b, payload_b
        )

    def test_diff_threshold_gives_diff_hash(self):
        a = _StrictnessZero(threshold=0.3)
        b = _StrictnessZero(threshold=0.5)
        fields = ("threshold", "evaluation_model", "strict_mode")
        pa = {f: getattr(a, f, None) for f in fields}
        pb = {f: getattr(b, f, None) for f in fields}
        assert _compute_metric_hash(a, pa) != _compute_metric_hash(b, pb)

    def test_diff_implementation_gives_diff_hash_custom(self):
        # _StrictnessZero and _StrictnessZeroV2 share public field names
        # and defaults but have different bodies. Source hashing must
        # distinguish them — this is the whole value prop of #3047.
        a = _StrictnessZero()
        b = _StrictnessZeroV2()
        fields = ("threshold", "evaluation_model", "strict_mode")
        pa = {f: getattr(a, f, None) for f in fields}
        pb = {f: getattr(b, f, None) for f in fields}
        assert _compute_metric_hash(a, pa) != _compute_metric_hash(b, pb)

    def test_async_methods_also_covered(self):
        """If we only hashed `measure`, swapping the a_measure body would
        still produce stale cache hits for async evaluations."""
        a = _CustomWithAsync()

        # Build a cloned variant that is byte-for-byte identical except
        # for a comment-free a_measure that returns a different score.
        class _CustomWithAsyncV2(BaseMetric):
            name = "_CustomWithAsync"

            def __init__(self, threshold: float = 0.5):
                self.threshold = threshold
                self.score = None
                self.error = None

            def measure(self, test_case, *args, **kwargs):
                self.score = 0.7

            async def a_measure(self, test_case, *args, **kwargs):
                self.score = 0.95

            def is_successful(self) -> bool:
                return self.score >= self.threshold

        b = _CustomWithAsyncV2()
        fields = ("threshold", "evaluation_model", "strict_mode")
        pa = {f: getattr(a, f, None) for f in fields}
        pb = {f: getattr(b, f, None) for f in fields}
        assert _compute_metric_hash(a, pa) != _compute_metric_hash(b, pb)


# ---------------------------------------------------------------------------
# same_metric_configs equivalence relation
# ---------------------------------------------------------------------------


class TestSameMetricConfigs:
    def test_identical_custom_metric_matches(self):
        m1 = _StrictnessZero()
        m2 = _StrictnessZero()
        cached = Cache.create_metric_configuration(m1)
        assert Cache.same_metric_configs(m2, cached) is True

    def test_source_change_invalidates_cache_hit(self):
        """The bug #3047 specifically calls out this scenario: a user
        tweaks their metric's `measure` method, re-runs, and the stale
        score must NOT be served from cache."""
        baseline = _StrictnessZero()
        after_edit = _StrictnessZeroV2()
        cached = Cache.create_metric_configuration(baseline)
        assert Cache.same_metric_configs(after_edit, cached) is False

    def test_param_change_invalidates_cache_hit(self):
        m1 = _StrictnessZero(threshold=0.4)
        m2 = _StrictnessZero(threshold=0.8)
        cached = Cache.create_metric_configuration(m1)
        assert Cache.same_metric_configs(m2, cached) is False

    def test_cross_class_names_never_match(self):
        a = _StrictnessZero()
        b = _CustomWithAsync()
        cached_a = Cache.create_metric_configuration(a)
        assert Cache.same_metric_configs(b, cached_a) is False

    def test_legacy_cache_without_hash_falls_back_to_field_check(self):
        # `_StrictnessZero` is a BaseMetric subclass with the default
        # include_reason=False and no language field. In main-branch
        # deepeval the disk-serialized copy would contain exactly those
        # values — i.e. False for include_reason, None for the absent
        # fields.
        m = _StrictnessZero(threshold=0.5)
        legacy_cfg = MetricConfiguration(
            threshold=0.5,
            strict_mode=False,
            evaluation_model=None,
            include_reason=False,
            n=None,
            criteria=None,
            language=None,
            embeddings=None,
            evaluation_steps=None,
            assessment_questions=None,
            evaluation_params=None,
        )
        assert Cache.same_metric_configs(m, legacy_cfg) is True

    def test_legacy_cache_different_threshold_rejected(self):
        m = _StrictnessZero(threshold=0.8)
        legacy_cfg = MetricConfiguration(
            threshold=0.5,
            strict_mode=False,
            evaluation_model=None,
            include_reason=False,
            n=None,
            criteria=None,
            language=None,
            embeddings=None,
            evaluation_steps=None,
            assessment_questions=None,
            evaluation_params=None,
        )
        assert Cache.same_metric_configs(m, legacy_cfg) is False


# ---------------------------------------------------------------------------
# Integration: Cache.get_metric_data round-trip
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    @staticmethod
    def _build_cached_test_case(metric: BaseMetric) -> CachedTestCase:
        metric_data = MetricData(
            name=metric.__name__,
            success=True,
            score=1.0,
            threshold=metric.threshold,
            reason="fixture",
            evaluation_model=None,
            evaluation_cost=0.0,
            verbose_logs=None,
            strict_mode=False,
            error=None,
            metric_metadata=None,
        )
        cached_metric_data = CachedMetricData(
            metric_data=metric_data,
            metric_configuration=Cache.create_metric_configuration(metric),
        )
        return CachedTestCase(cached_metrics_data=[cached_metric_data])

    def test_roundtrip_custom_metric_same_class(self):
        writer_cls = _StrictnessZero()
        reader_cls = _StrictnessZero()
        cached = self._build_cached_test_case(writer_cls)
        assert Cache.get_metric_data(reader_cls, cached) is not None

    def test_roundtrip_custom_metric_source_edit_rejects_hit(self):
        """End-to-end mirror of the issue: the user re-defines `measure`
        and a previously cached score must no longer be returned."""
        before = _StrictnessZero()
        after = _StrictnessZeroV2()
        cached = self._build_cached_test_case(before)
        assert Cache.get_metric_data(after, cached) is None

    def test_roundtrip_param_change_rejects_hit(self):
        before = _StrictnessZero(threshold=0.3)
        after = _StrictnessZero(threshold=0.9)
        cached = self._build_cached_test_case(before)
        assert Cache.get_metric_data(after, cached) is None
