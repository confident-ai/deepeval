"""Tests for trimAndLoadJson None-guard (issue #2554)."""

import pytest

from deepeval.metrics.utils import trimAndLoadJson as metrics_trimAndLoadJson
from deepeval.dataset.utils import trimAndLoadJson as dataset_trimAndLoadJson


class TestMetricsTrimAndLoadJson:
    def test_none_input_raises_value_error(self):
        with pytest.raises(ValueError, match="returned None"):
            metrics_trimAndLoadJson(None)

    def test_none_input_sets_metric_error(self):
        class FakeMetric:
            error = None

        metric = FakeMetric()
        with pytest.raises(ValueError, match="returned None"):
            metrics_trimAndLoadJson(None, metric=metric)
        assert metric.error is not None

    def test_valid_json(self):
        result = metrics_trimAndLoadJson('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        result = metrics_trimAndLoadJson('some text {"key": 1} more text')
        assert result == {"key": 1}

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError):
            metrics_trimAndLoadJson("not json at all")


class TestDatasetTrimAndLoadJson:
    def test_none_input_raises_value_error(self):
        with pytest.raises(ValueError, match="returned None"):
            dataset_trimAndLoadJson(None)

    def test_valid_json(self):
        result = dataset_trimAndLoadJson('{"key": "value"}')
        assert result == {"key": "value"}
