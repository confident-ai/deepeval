import pytest

from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig, DisplayConfig
from deepeval.metrics import (
    GEval,
    PatternMatchMetric,
    ToolPermissionMetric,
)
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.test_run.cache import Cache, MetricConfiguration


class _StaticJudge(DeepEvalBaseLLM):
    def __init__(self):
        pass

    def load_model(self):
        return self

    def get_model_name(self):
        return "static-judge"

    def generate(self, prompt, schema=None):
        raise AssertionError("the cache comparison must not call the judge")

    async def a_generate(self, prompt, schema=None):
        raise AssertionError("the cache comparison must not call the judge")


LENIENT = [Rubric(score_range=(0, 10), expected_outcome="Any helpful reply.")]
STRICT = [
    Rubric(score_range=(0, 2), expected_outcome="No policy link."),
    Rubric(score_range=(3, 10), expected_outcome="Links the policy."),
]


def _geval(rubric=None):
    return GEval(
        name="Refund handling",
        criteria="Does the reply handle the refund request well?",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        rubric=rubric,
        model=_StaticJudge(),
    )


def _as_read_from_cache_file(configuration: MetricConfiguration):
    return MetricConfiguration.model_validate_json(
        configuration.model_dump_json()
    )


def test_changed_geval_rubric_does_not_reuse_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval(LENIENT))
    )

    assert not Cache.same_metric_configs(_geval(STRICT), cached)
    assert not Cache.same_metric_configs(_geval(), cached)


def test_same_geval_rubric_reuses_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval(STRICT))
    )

    assert Cache.same_metric_configs(_geval(STRICT), cached)


def test_geval_without_rubric_reuses_cached_score():
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval())
    )

    assert Cache.same_metric_configs(_geval(), cached)


def test_rubric_added_after_caching_does_not_reuse_cached_score():
    # Cache files written before the rubric was recorded look the same: the
    # entry has no rubric.
    cached = _as_read_from_cache_file(
        Cache.create_metric_configuration(_geval())
    )
    assert getattr(cached, "rubric", None) is None

    assert not Cache.same_metric_configs(_geval(STRICT), cached)


# Deterministic metrics: the score depends only on these settings, so a
# cached score can only be reused for the same ones.


def _cached(metric):
    return _as_read_from_cache_file(Cache.create_metric_configuration(metric))


@pytest.mark.parametrize(
    "cached_metric, metric",
    [
        (PatternMatchMetric(pattern=r".*"), PatternMatchMetric(pattern=r"\d+")),
        (
            PatternMatchMetric(pattern="ok", ignore_case=True),
            PatternMatchMetric(pattern="ok", ignore_case=False),
        ),
        (
            ToolPermissionMetric(denied_tools=["send_email"]),
            ToolPermissionMetric(denied_tools=["send_email", "delete_db"]),
        ),
        (
            ToolPermissionMetric(allowed_tools=["search", "delete_db"]),
            ToolPermissionMetric(allowed_tools=["search"]),
        ),
        (
            ToolPermissionMetric(allowed_tools=[], denied_tools=["delete_db"]),
            ToolPermissionMetric(denied_tools=["delete_db"]),
        ),
    ],
    ids=[
        "pattern",
        "ignore_case",
        "denied_tools",
        "allowed_tools",
        "empty_allowlist_vs_none",
    ],
)
def test_changed_deterministic_metric_settings_do_not_reuse_cached_score(
    cached_metric, metric
):
    assert not Cache.same_metric_configs(metric, _cached(cached_metric))


@pytest.mark.parametrize(
    "cached_metric, metric",
    [
        (
            PatternMatchMetric(pattern=r"\d+"),
            PatternMatchMetric(pattern=r"\d+"),
        ),
        (
            ToolPermissionMetric(
                allowed_tools=["search", "lookup"], denied_tools=["a", "b"]
            ),
            ToolPermissionMetric(
                allowed_tools=["lookup", "search"], denied_tools=["b", "a"]
            ),
        ),
    ],
    ids=["pattern", "tool_lists_in_another_order"],
)
def test_same_deterministic_metric_settings_reuse_cached_score(
    cached_metric, metric
):
    assert Cache.same_metric_configs(metric, _cached(cached_metric))


def test_tightened_pattern_is_rechecked_by_evaluate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def run(pattern):
        result = evaluate(
            [LLMTestCase(input="Order id?", actual_output="I don't know")],
            [PatternMatchMetric(pattern=pattern)],
            cache_config=CacheConfig(use_cache=True, write_cache=True),
            display_config=DisplayConfig(
                print_results=False, show_indicator=False
            ),
        )
        return result.test_results[0].metrics_data[0].success

    assert run(r".*") is True
    assert run(r"\d+") is False
