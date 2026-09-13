import pytest

from deepeval import assert_test, evaluate
from deepeval.evaluate.utils import create_metric_data
from deepeval.metrics import (
    BaseConversationalMetric,
    BaseMetric,
    ExactMatchMetric,
    ToolPermissionMetric,
)
from deepeval.metrics.utils import (
    check_at_least_one_metric_has_threshold,
    copy_metrics,
)
from deepeval.test_case import LLMTestCase
from deepeval.tracing.api import MetricData


class StubMetric(BaseMetric):
    threshold = 0.5

    def measure(self, test_case, *args, **kwargs):
        return 1

    async def a_measure(self, test_case, *args, **kwargs):
        return 1


class StubConversationalMetric(BaseConversationalMetric):
    threshold = 0.5

    def measure(self, test_case, *args, **kwargs):
        return 1

    async def a_measure(self, test_case, *args, **kwargs):
        return 1


class NarrowToolPermissionMetric(ToolPermissionMetric):
    def __init__(self, threshold=1.0):
        super().__init__(denied_tools=["shell"], threshold=threshold)


class KwargsExactMatchMetric(ExactMatchMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NarrowConversationalMetric(BaseConversationalMetric):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.custom_config = {"labels": ["conversation"]}

    def measure(self, test_case, *args, **kwargs):
        return 1

    async def a_measure(self, test_case, *args, **kwargs):
        return 1


class FixedThresholdMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.8

    def measure(self, test_case, *args, **kwargs):
        return 1

    async def a_measure(self, test_case, *args, **kwargs):
        return 1


def test_metric_data_threshold_defaults_to_none():
    metric_data = MetricData(name="Metric")

    assert metric_data.threshold is None
    assert metric_data.success is None


def test_metric_constructor_accepts_none_threshold_and_flaky():
    metric = ExactMatchMetric(threshold=None, flaky=True)
    test_case = LLMTestCase(
        input="input",
        actual_output="output",
        expected_output="output",
    )

    assert metric.threshold is None
    assert metric.flaky is True

    metric.measure(test_case, _show_indicator=False)
    assert metric.score == 1
    assert metric.success is None

    metric_data = create_metric_data(metric)
    assert metric_data.threshold is None
    assert metric_data.success is None
    assert metric_data.flaky is True


def test_copy_metrics_preserves_flaky():
    metric = ExactMatchMetric(flaky=True)

    copied_metric = copy_metrics([metric])[0]

    assert copied_metric.flaky is True


@pytest.mark.parametrize(
    "metric",
    [
        NarrowToolPermissionMetric(),
        KwargsExactMatchMetric(threshold=0.8),
        NarrowConversationalMetric(),
        FixedThresholdMetric(),
    ],
)
def test_copy_metrics_clones_custom_metrics_without_reconstructing(
    metric,
):
    metric.custom_config = ["custom"]
    metric.model = object()
    metric.score = 0.25
    metric.reason = "old reason"
    metric.success = False
    metric.error = "old error"
    metric.verdicts = ["old verdict"]
    metric.skipped = True
    metric.evaluation_cost = 3.0
    metric.input_tokens = 4
    metric.output_tokens = 5
    metric.verbose_logs = "old logs"

    copied_metric = copy_metrics([metric])[0]

    assert type(copied_metric) is type(metric)
    assert copied_metric is not metric
    assert copied_metric.custom_config == metric.custom_config
    assert copied_metric.custom_config is not metric.custom_config
    assert copied_metric.model is metric.model
    assert copied_metric.threshold == metric.threshold
    for field in (
        "score",
        "reason",
        "success",
        "error",
        "verdicts",
        "skipped",
        "evaluation_cost",
        "input_tokens",
        "output_tokens",
        "verbose_logs",
    ):
        assert getattr(copied_metric, field) is None


@pytest.mark.parametrize("metric_class", [StubMetric, StubConversationalMetric])
@pytest.mark.parametrize("error", [None, "Metric failed"])
def test_none_threshold_is_reflected_in_metric_data(metric_class, error):
    metric = metric_class()
    metric.threshold = None
    metric.error = error

    metric_data = create_metric_data(metric)

    assert metric_data.threshold is None
    assert metric_data.success is None


@pytest.mark.parametrize("metric_class", [StubMetric, StubConversationalMetric])
def test_flaky_defaults_to_false_and_is_reflected_in_metric_data(metric_class):
    metric = metric_class()

    assert metric.flaky is False
    assert create_metric_data(metric).flaky is False


@pytest.mark.parametrize("metric_class", [StubMetric, StubConversationalMetric])
@pytest.mark.parametrize("error", [None, "Metric failed"])
def test_flaky_is_reflected_in_metric_data(metric_class, error):
    metric = metric_class()
    metric.flaky = True
    metric.error = error

    metric_data = create_metric_data(metric)

    assert metric_data.flaky is True
    assert metric_data.model_dump(by_alias=True)["flaky"] is True


def test_check_at_least_one_metric_has_threshold():
    # Passes as long as one non-flaky metric has a threshold
    check_at_least_one_metric_has_threshold(
        [
            ExactMatchMetric(threshold=None),
            ExactMatchMetric(threshold=0.5, flaky=True),
            ExactMatchMetric(),
        ]
    )

    # No threshold at all
    with pytest.raises(ValueError, match="non-flaky metric"):
        check_at_least_one_metric_has_threshold(
            [ExactMatchMetric(threshold=None)]
        )

    # Has a threshold, but the only candidate is flaky
    with pytest.raises(ValueError, match="non-flaky metric"):
        check_at_least_one_metric_has_threshold(
            [ExactMatchMetric(threshold=0.5, flaky=True)]
        )


def test_evaluate_requires_at_least_one_metric_threshold():
    test_case = LLMTestCase(input="input", actual_output="output")

    with pytest.raises(ValueError, match="non-flaky metric"):
        evaluate(
            test_cases=[test_case],
            metrics=[ExactMatchMetric(threshold=None)],
        )


def test_assert_test_requires_at_least_one_metric_threshold():
    test_case = LLMTestCase(input="input", actual_output="output")

    with pytest.raises(ValueError, match="non-flaky metric"):
        assert_test(
            test_case=test_case,
            metrics=[ExactMatchMetric(threshold=None)],
        )
