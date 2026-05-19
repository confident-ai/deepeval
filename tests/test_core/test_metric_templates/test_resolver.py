"""Tests for deepeval.metric_templates.resolver."""

from __future__ import annotations

import pytest

from deepeval.metric_templates.resolver import (
    MetricTemplateInterpolationError,
    MetricTemplateNotFoundError,
    clear_metric_template_cache,
    get_base_template,
    get_raw_template,
    iter_base_template_methods,
    resolve_template,
)


@pytest.fixture(autouse=True)
def _clear_template_cache():
    clear_metric_template_cache()
    yield
    clear_metric_template_cache()


def test_get_raw_template_known_pair():
    s = get_base_template("FaithfulnessMetric", "generate_claims")
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


def test_iter_base_template_methods():
    pairs = iter_base_template_methods("TaskNode")
    methods = [m for m, _ in pairs]
    assert "generate_task_output" in methods
    assert all(isinstance(t, str) and t for _, t in pairs)


def test_resolve_template_fragment_and_kwargs():
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


def test_resolve_template_branching_optional_multimodal():
    raw = get_base_template("AnswerRelevancyMetric", "generate_statements")
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


def test_resolve_template_strict_unresolved():
    with pytest.raises(MetricTemplateInterpolationError) as ei:
        resolve_template(
            "AnswerRelevancyMetric",
            "generate_statements",
            multimodal=False,
        )
    assert "actual_output" in str(ei.value) or ei.value.unresolved


def test_resolve_template_label_pass_fail():
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


def test_get_raw_template_uses_community_bundle(monkeypatch):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "get_active_metric_template_language", lambda: "hindi")
    monkeypatch.setattr(
        res.TemplateRegistry,
        "get_custom_templates",
        lambda self, slug: {},
    )
    monkeypatch.setattr(
        res.TemplateRegistry,
        "get_community_templates",
        lambda self, slug: {
            "FaithfulnessMetric": {"generate_claims": "HINDI_CLAIMS_TEMPLATE"},
        },
    )
    assert (
        get_raw_template("FaithfulnessMetric", "generate_claims")
        == "HINDI_CLAIMS_TEMPLATE"
    )


def test_get_raw_template_uses_custom_over_community(monkeypatch):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "get_active_metric_template_language", lambda: "hindi")
    monkeypatch.setattr(
        res.TemplateRegistry,
        "get_custom_templates",
        lambda self, slug: {
            "FaithfulnessMetric": {"generate_claims": "CUSTOM_CLAIMS"},
        },
    )
    monkeypatch.setattr(
        res.TemplateRegistry,
        "get_community_templates",
        lambda self, slug: {
            "FaithfulnessMetric": {"generate_claims": "COMMUNITY_CLAIMS"},
        },
    )
    assert get_raw_template("FaithfulnessMetric", "generate_claims") == "CUSTOM_CLAIMS"


def test_get_raw_template_falls_back_to_english_when_missing_key(monkeypatch, capsys):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "get_active_metric_template_language", lambda: "hindi")
    monkeypatch.setattr(
        res.TemplateRegistry,
        "get_custom_templates",
        lambda self, slug: {},
    )
    monkeypatch.setattr(
        res.TemplateRegistry,
        "get_community_templates",
        lambda self, slug: {"FaithfulnessMetric": {"generate_claims": "HINDI_ONLY"}},
    )

    bundle = get_base_template("FaithfulnessMetric", "generate_truths")
    assert get_raw_template("FaithfulnessMetric", "generate_truths") == bundle
    get_raw_template("FaithfulnessMetric", "generate_truths")

    out = capsys.readouterr().out
    assert out.count("No 'hindi' translation found") == 1
    assert "FaithfulnessMetric" in out


def test_get_raw_template_uses_english_when_language_unset(monkeypatch):
    import deepeval.metric_templates.resolver as res

    monkeypatch.setattr(res, "get_active_metric_template_language", lambda: None)
    assert (
        get_raw_template("FaithfulnessMetric", "generate_claims")
        == get_base_template("FaithfulnessMetric", "generate_claims")
    )
