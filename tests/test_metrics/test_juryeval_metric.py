import pytest

from deepeval.test_case import LLMTestCase


class TestJuryEvalMetric:
    def test_juryeval_not_installed_raises(self):
        try:
            import juryeval  # noqa: F401
            pytest.skip("juryeval is installed")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="juryeval is required"):
            from deepeval.metrics.juryeval import JuryEvalMetric

            JuryEvalMetric(metric_fn=lambda preds, refs: 0.5)

    def test_custom_metric_fn(self):
        from deepeval.metrics.juryeval import JuryEvalMetric

        def dummy_fn(predictions, references=None, **kwargs):
            return 0.75

        metric = JuryEvalMetric(metric_fn=dummy_fn, metric_name="Custom", threshold=0.5)
        test_case = LLMTestCase(
            input="test input",
            actual_output="test output",
        )
        score = metric.measure(test_case)
        assert score == 0.75
        assert metric.score == 0.75
        assert metric.is_successful() is True
        assert "Custom" in metric.__name__

    def test_custom_metric_fn_below_threshold(self):
        from deepeval.metrics.juryeval import JuryEvalMetric

        def dummy_fn(predictions, references=None, **kwargs):
            return 0.3

        metric = JuryEvalMetric(metric_fn=dummy_fn, metric_name="Custom", threshold=0.5)
        test_case = LLMTestCase(
            input="test input",
            actual_output="test output",
        )
        metric.measure(test_case)
        assert metric.is_successful() is False

    def test_async_measure(self):
        from deepeval.metrics.juryeval import JuryEvalMetric

        def dummy_fn(predictions, references=None, **kwargs):
            return 0.8

        metric = JuryEvalMetric(metric_fn=dummy_fn, metric_name="Custom")
        test_case = LLMTestCase(
            input="test input",
            actual_output="test output",
        )
        import asyncio

        score = asyncio.run(metric.a_measure(test_case))
        assert score == 0.8

    def test_with_expected_output(self):
        from deepeval.metrics.juryeval import JuryEvalMetric

        def dummy_fn(predictions, references=None, **kwargs):
            assert references is not None
            return 1.0

        metric = JuryEvalMetric(metric_fn=dummy_fn, metric_name="Custom")
        test_case = LLMTestCase(
            input="test input",
            actual_output="test output",
            expected_output="expected output",
        )
        score = metric.measure(test_case)
        assert score == 1.0
