import pytest

from deepeval import evaluate
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.evaluate.types import EvaluationResult
from deepeval.errors import MissingTestCaseParamsError
from deepeval.test_case import LLMTestCase

from .helpers import (
    DeterministicFailingMetric,
    DeterministicPassingMetric,
    DeterministicRaisingMetric,
    DeterministicRequiresRetrievalContextMetric,
    build_llm_test_cases_from_goldens,
    build_single_turn_dataset,
)

# ===========================================================================
# ErrorConfig: missing params behavior
# ===========================================================================


def test_error_config_missing_params_raises_by_default():
    """By default, missing required test case params raises MissingTestCaseParamsError."""
    # Create a test case without retrieval_context
    test_case = LLMTestCase(
        input="What is your name?",
        actual_output="My name is DeepEval.",
        retrieval_context=None,  # Missing required param
    )

    with pytest.raises(MissingTestCaseParamsError):
        evaluate(
            test_cases=[test_case],
            metrics=[DeterministicRequiresRetrievalContextMetric()],
            hyperparameters={"model": "offline-stub"},
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            display_config=DisplayConfig(
                show_indicator=False, print_results=False
            ),
            error_config=ErrorConfig(skip_on_missing_params=False),
        )


def test_error_config_skip_on_missing_params_skips_metric():
    """When skip_on_missing_params=True, metrics with missing required params are skipped."""
    # Create a test case without retrieval_context
    test_case = LLMTestCase(
        input="What is your name?",
        actual_output="My name is DeepEval.",
        retrieval_context=None,  # Missing required param
    )

    result = evaluate(
        test_cases=[test_case],
        metrics=[DeterministicRequiresRetrievalContextMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(skip_on_missing_params=True),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 1

    tr = result.test_results[0]
    # When metric is skipped, it should not appear in metrics_data
    assert (
        len(tr.metrics_data) == 0
    ), "Skipped metric should not appear in metrics_data"


def test_error_config_skip_on_missing_params_does_not_skip_when_complete():
    """When skip_on_missing_params=True, a complete test case is still evaluated."""
    # Create a test case with retrieval_context
    test_case = LLMTestCase(
        input="What is your name?",
        actual_output="My name is DeepEval.",
        retrieval_context=["context chunk 1", "context chunk 2"],
    )

    result = evaluate(
        test_cases=[test_case],
        metrics=[DeterministicRequiresRetrievalContextMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(skip_on_missing_params=True),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 1

    tr = result.test_results[0]
    # Metric should be evaluated and present
    assert len(tr.metrics_data) == 1
    assert tr.metrics_data[0].success is True


def test_error_config_skip_on_missing_params_takes_precedence_over_ignore_errors():
    """skip_on_missing_params=True takes precedence over ignore_errors=True when params are missing."""
    test_case = LLMTestCase(
        input="What is your name?",
        actual_output="My name is DeepEval.",
        retrieval_context=None,  # Missing required param
    )

    result = evaluate(
        test_cases=[test_case],
        metrics=[DeterministicRequiresRetrievalContextMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(
            skip_on_missing_params=True,
            ignore_errors=True,
        ),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 1

    tr = result.test_results[0]
    # skip_on_missing_params takes precedence: metric should be skipped entirely
    # (not present in metrics_data), rather than showing as ignored error
    assert len(tr.metrics_data) == 0, (
        "skip_on_missing_params should take precedence: metric should be "
        "skipped (absent from metrics_data), not shown as ignored error"
    )


# -----------------------------------------------------------------------------
# ErrorConfig: ignore_errors behavior
# -----------------------------------------------------------------------------


def test_error_config_ignore_errors_raises_by_default():
    """By default, exceptions raised by metrics propagate (ignore_errors=False)."""
    test_case = LLMTestCase(
        input="What is your name?",
        actual_output="My name is DeepEval.",
    )

    with pytest.raises(RuntimeError, match="always raises"):
        evaluate(
            test_cases=[test_case],
            metrics=[DeterministicRaisingMetric()],
            hyperparameters={"model": "offline-stub"},
            async_config=AsyncConfig(run_async=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
            display_config=DisplayConfig(
                show_indicator=False, print_results=False
            ),
            error_config=ErrorConfig(ignore_errors=False),
        )


def test_error_config_ignore_errors_captures_metric_exception():
    """When ignore_errors=True, metric exceptions are captured and surfaced in the result."""
    test_case = LLMTestCase(
        input="What is your name?",
        actual_output="My name is DeepEval.",
    )

    result = evaluate(
        test_cases=[test_case],
        metrics=[DeterministicRaisingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(ignore_errors=True),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 1

    tr = result.test_results[0]
    assert len(tr.metrics_data) == 1

    md = tr.metrics_data[0]
    # When error is ignored, metric should be marked as failed
    assert md.success is False
    # Error message should be captured
    assert md.error is not None
    assert "always raises" in md.error


def test_error_config_ignore_errors_does_not_affect_passing_metrics():
    """ignore_errors=True only affects metrics that raise; passing metrics remain unaffected."""
    test_case = LLMTestCase(
        input="What is your name?",
        actual_output="My name is DeepEval.",
    )

    result = evaluate(
        test_cases=[test_case],
        metrics=[
            DeterministicPassingMetric(),
            DeterministicRaisingMetric(),
        ],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        error_config=ErrorConfig(ignore_errors=True),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == 1

    tr = result.test_results[0]
    assert len(tr.metrics_data) == 2

    # Find each metric's data by name
    passing_md = next(
        md for md in tr.metrics_data if md.name == "DeterministicPassingMetric"
    )
    raising_md = next(
        md for md in tr.metrics_data if md.name == "DeterministicRaisingMetric"
    )

    # Passing metric should succeed
    assert passing_md.success is True
    assert passing_md.error is None

    # Raising metric should fail with captured error
    assert raising_md.success is False
    assert raising_md.error is not None


# -----------------------------------------------------------------------------
# AsyncConfig behavior
# -----------------------------------------------------------------------------


def test_async_config_sync_and_async_produce_equivalent_results():
    """Sync and async evaluation produce equivalent results for deterministic metrics."""
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    # Run with run_async=False (sync)
    result_sync = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # Run with run_async=True (async)
    result_async = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(
            run_async=True, max_concurrent=1, throttle_value=0
        ),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # Both should return valid results
    assert isinstance(result_sync, EvaluationResult)
    assert isinstance(result_async, EvaluationResult)

    # Same number of test results
    assert len(result_sync.test_results) == len(result_async.test_results)

    # Each test result should have same success status
    for tr_sync, tr_async in zip(
        result_sync.test_results, result_async.test_results
    ):
        assert tr_sync.success == tr_async.success
        assert len(tr_sync.metrics_data) == len(tr_async.metrics_data)

        # Metric names and success should match
        for md_sync, md_async in zip(
            tr_sync.metrics_data, tr_async.metrics_data
        ):
            assert md_sync.name == md_async.name
            assert md_sync.success == md_async.success


def test_async_config_max_concurrent_must_be_positive():
    """max_concurrent must be >= 1."""
    with pytest.raises(ValueError, match="max_concurrent"):
        AsyncConfig(max_concurrent=0)


def test_async_config_throttle_value_must_be_non_negative():
    """throttle_value must be >= 0."""
    with pytest.raises(ValueError, match="throttle_value"):
        AsyncConfig(throttle_value=-1)


# -----------------------------------------------------------------------------
# CacheConfig behavior
# -----------------------------------------------------------------------------


def test_cache_config_disabled_does_not_break_evaluate():
    """Evaluation succeeds when caching is fully disabled."""
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)

    for tr in result.test_results:
        assert tr.success is True
        assert len(tr.metrics_data) >= 1


# -----------------------------------------------------------------------------
# DisplayConfig behavior
# -----------------------------------------------------------------------------


def test_display_config_all_does_not_break_evaluate():
    """display_option=ALL does not affect evaluation execution."""
    from deepeval.test_run.test_run import TestRunResultDisplay

    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            display_option=TestRunResultDisplay.ALL,
        ),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)


def test_display_config_failing_does_not_break_evaluate():
    """display_option=FAILING does not affect evaluation execution."""
    from deepeval.test_run.test_run import TestRunResultDisplay

    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicFailingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            display_option=TestRunResultDisplay.FAILING,
        ),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)

    # All results should be failures (we used DeterministicFailingMetric)
    for tr in result.test_results:
        assert tr.success is False


def test_display_config_passing_does_not_break_evaluate():
    """display_option=PASSING does not affect evaluation execution."""
    from deepeval.test_run.test_run import TestRunResultDisplay

    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            display_option=TestRunResultDisplay.PASSING,
        ),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)

    # All results should be successes (we used DeterministicPassingMetric)
    for tr in result.test_results:
        assert tr.success is True


def test_display_config_does_not_affect_evaluation_results():
    """DisplayConfig options should not affect evaluation outcomes, only printing."""
    from deepeval.test_run.test_run import TestRunResultDisplay

    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    # Run with display="all"
    result_all = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            display_option=TestRunResultDisplay.ALL,
        ),
    )

    # Run with display="passing"
    result_passing = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            display_option=TestRunResultDisplay.PASSING,
        ),
    )

    # Results should be identical regardless of display option
    assert len(result_all.test_results) == len(result_passing.test_results)

    for tr_all, tr_pass in zip(
        result_all.test_results, result_passing.test_results
    ):
        assert tr_all.success == tr_pass.success
        assert len(tr_all.metrics_data) == len(tr_pass.metrics_data)
