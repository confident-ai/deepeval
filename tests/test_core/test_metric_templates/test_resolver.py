"""Tests for deepeval.metric_templates.resolver."""

from __future__ import annotations

import json

import pytest

from deepeval.metric_templates.resolver import (
    MetricTemplateInterpolationError,
    MetricTemplateNotFoundError,
    clear_metric_template_cache,
    get_bundle_only_template,
    get_raw_template,
    iter_bundle_template_methods,
    list_methods,
    resolve_template,
)


@pytest.fixture(autouse=True)
def _clear_template_cache():
    clear_metric_template_cache()
    yield
    clear_metric_template_cache()


def test_get_raw_template_known_pair():
    s = get_bundle_only_template("FaithfulnessMetric", "generate_claims")
    assert isinstance(s, str)
    assert len(s) > 50
    assert "{% if multimodal %}" in s or "claims" in s.lower()


def test_get_raw_template_missing_class():
    with pytest.raises(MetricTemplateNotFoundError) as ei:
        get_raw_template("NonexistentMetricClassXYZ", "generate_claims")
    assert "NonexistentMetricClassXYZ" in str(ei.value)


def test_get_raw_template_missing_method():
    with pytest.raises(MetricTemplateNotFoundError) as ei:
        get_raw_template("FaithfulnessMetric", "nonexistent_method_xyz")
    assert "nonexistent_method_xyz" in str(ei.value)
    assert "generate_claims" in str(ei.value)


def test_list_methods():
    names = list_methods("TaskNode")
    assert "generate_task_output" in names


def test_iter_bundle_template_methods():
    pairs = iter_bundle_template_methods("TaskNode")
    methods = [m for m, _ in pairs]
    assert "generate_task_output" in methods
    assert all(isinstance(t, str) and t for _, t in pairs)


def test_resolve_template_fragment_and_kwargs(tmp_path, monkeypatch):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "HIDDEN_DIR", str(tmp_path))
    clear_metric_template_cache()
    out = resolve_template(
        "TaskNode",
        "generate_task_output",
        multimodal=False,
        instructions="INS",
        text="TXT",
    )
    assert "INS" in out
    assert "TXT" in out
    assert "{{" not in out
    assert "MULTIMODAL INPUT RULES" in out


def test_resolve_template_branching_optional_multimodal(tmp_path, monkeypatch):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "HIDDEN_DIR", str(tmp_path))
    clear_metric_template_cache()
    raw = get_bundle_only_template("AnswerRelevancyMetric", "generate_statements")
    assert "{% if multimodal %}" in raw
    out_off = resolve_template(
        "AnswerRelevancyMetric",
        "generate_statements",
        multimodal=False,
        actual_output="hello",
    )
    assert "{% if multimodal %}" not in out_off
    assert "MULTIMODAL INPUT RULES" not in out_off

    out_on = resolve_template(
        "AnswerRelevancyMetric",
        "generate_statements",
        multimodal=True,
        actual_output="hello",
    )
    assert "MULTIMODAL INPUT RULES" in out_on


def test_resolve_template_strict_unresolved(tmp_path, monkeypatch):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "HIDDEN_DIR", str(tmp_path))
    clear_metric_template_cache()
    with pytest.raises(MetricTemplateInterpolationError) as ei:
        resolve_template(
            "AnswerRelevancyMetric",
            "generate_statements",
            multimodal=False,
        )
    assert "actual_output" in str(ei.value) or ei.value.unresolved


def test_resolve_template_label_pass_fail(tmp_path, monkeypatch):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "HIDDEN_DIR", str(tmp_path))
    clear_metric_template_cache()
    out = resolve_template(
        "GoalAccuracyMetric",
        "get_final_reason",
        multimodal=False,
        goal_evaluations="{}",
        plan_evalautions="{}",
        final_score=0.9,
        threshold=0.5,
    )
    assert "PASS" in out
    assert "{{" not in out


def test_get_bundle_only_template_ignores_hidden(tmp_path, monkeypatch):
    import deepeval.metric_templates.resolver as res

    hidden_path = tmp_path / "templates.json"
    hidden_path.write_text(
        json.dumps({"FaithfulnessMetric": {"generate_claims": "HIDDEN_ONLY"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(res, "HIDDEN_DIR", str(tmp_path))
    clear_metric_template_cache()
    overridden = get_raw_template("FaithfulnessMetric", "generate_claims")
    bundle_text = get_bundle_only_template("FaithfulnessMetric", "generate_claims")
    assert overridden == "HIDDEN_ONLY"
    assert bundle_text != "HIDDEN_ONLY"
    assert len(bundle_text) > 50
