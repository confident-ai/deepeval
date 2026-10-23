import pytest

from deepeval.evaluate.utils import create_metric_data
from deepeval.metrics import ExactMatchMetric, ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.test_run.api import MetricData
from deepeval.test_run.cache import (
    Cache,
    CachedMetricData,
    CachedTestCase,
    MetricConfiguration,
)


class OfflineModel(DeepEvalBaseLLM):
    def load_model(self):
        return None

    def generate(self, *args, **kwargs):
        raise AssertionError("These cache tests must not call a judge")

    async def a_generate(self, *args, **kwargs):
        raise AssertionError("These cache tests must not call a judge")

    def get_model_name(self):
        return "offline-cache-test"


def make_metric(**kwargs):
    return ToolCorrectnessMetric(
        model=OfflineModel(), async_mode=False, **kwargs
    )


def cached_case(metric, metric_data=None):
    cached = CachedTestCase(
        cached_metrics_data=[
            CachedMetricData(
                metric_data=metric_data
                or MetricData(name=metric.__name__, score=1),
                metric_configuration=Cache.create_metric_configuration(metric),
            )
        ]
    )
    return CachedTestCase.model_validate_json(cached.model_dump_json())


@pytest.mark.parametrize(
    "options,tools_called,expected_score",
    [
        (
            {"should_exact_match": True},
            [
                ToolCall(name="first"),
                ToolCall(name="second"),
                ToolCall(name="extra"),
            ],
            0,
        ),
        (
            {"should_consider_ordering": True},
            [ToolCall(name="second"), ToolCall(name="first")],
            0.5,
        ),
    ],
)
def test_changed_matching_rule_does_not_reuse_a_different_score(
    options, tools_called, expected_score
):
    test_case = LLMTestCase(
        input="Run the two tools in order",
        actual_output="Done",
        tools_called=tools_called,
        expected_tools=[ToolCall(name="first"), ToolCall(name="second")],
    )
    original = make_metric()
    original.measure(test_case, _show_indicator=False)
    assert original.score == 1
    cached = cached_case(original, create_metric_data(original))
    assert Cache.get_metric_data(original, cached).metric_data.score == 1

    changed = make_metric(**options)
    changed.measure(test_case, _show_indicator=False)
    assert changed.score == expected_score
    assert Cache.get_metric_data(changed, cached) is None


@pytest.mark.parametrize(
    "original_tools,new_tools",
    [
        (None, [ToolCall(name="search")]),
        ([ToolCall(name="search")], None),
        ([ToolCall(name="search")], [ToolCall(name="lookup")]),
        (
            [ToolCall(name="search", input_parameters={"query": "old"})],
            [ToolCall(name="search", input_parameters={"query": "new"})],
        ),
    ],
)
def test_changed_available_tools_invalidate_cached_metric(
    original_tools, new_tools
):
    cached = cached_case(make_metric(available_tools=original_tools))
    assert (
        Cache.get_metric_data(make_metric(available_tools=new_tools), cached)
        is None
    )


def test_unchanged_tool_configuration_survives_cache_json_round_trip():
    options = {
        "should_exact_match": True,
        "should_consider_ordering": True,
        "available_tools": [
            ToolCall(name="search", input_parameters={"query": "hello"})
        ],
    }
    cached = cached_case(make_metric(**options))
    assert Cache.get_metric_data(make_metric(**options), cached) is not None


def test_legacy_tool_configuration_is_a_safe_cache_miss():
    metric = make_metric()
    legacy = Cache.create_metric_configuration(metric).model_dump(
        exclude={
            "should_exact_match",
            "should_consider_ordering",
            "available_tools",
        }
    )
    assert not Cache.same_metric_configs(
        metric, MetricConfiguration.model_validate(legacy)
    )


def test_legacy_configuration_remains_usable_for_other_metrics():
    metric = ExactMatchMetric()
    legacy = Cache.create_metric_configuration(metric).model_dump(
        exclude={
            "should_exact_match",
            "should_consider_ordering",
            "available_tools",
        }
    )
    assert Cache.same_metric_configs(
        metric, MetricConfiguration.model_validate(legacy)
    )
