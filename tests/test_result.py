from deepeval.run_test import TestResult


def test_result():
    result = TestResult(
        success=True,
        score=2.0,
        metric_name="test_metric",
        query="test_query",
        output="test_output",
        expected_output="test_expected_output",
        metadata=None,
        context="test_context",
    )
    assert result.score == 1.0
