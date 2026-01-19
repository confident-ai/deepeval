import json
import os
import subprocess
import pytest

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import (
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.errors import MissingTestCaseParamsError
from deepeval.test_case import LLMTestCase

from .helpers import (
    DeterministicContainsExpectedOutputMetric,
    DeterministicConversationalMetric,
    DeterministicFailingMetric,
    DeterministicPassingMetric,
    DeterministicRaisingMetric,
    DeterministicRequiresRetrievalContextMetric,
    build_llm_test_cases_from_goldens,
    build_single_turn_dataset,
    build_multi_turn_dataset,
    build_conversational_test_cases_manually,
    save_dataset_as_csv_and_load,
    save_dataset_as_json_and_load,
)


def test_docs_single_turn_e2e_python_api_result_shape_and_dataset_json_artifact(
    tmp_path,
):
    """
    Doc-driven regression test for:
      - dataset creation -> test case creation -> evaluate() -> result object
      - dataset JSON artifact schema (Option A)

    Deterministic/offline: uses a deterministic custom metric (no network).
    """
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicContainsExpectedOutputMetric()],
        hyperparameters={"model": "offline-stub", "system_prompt": "offline"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # Verify result is the expected type with correct structure
    assert isinstance(result, EvaluationResult)
    assert isinstance(result.test_results, list)
    assert len(result.test_results) == len(test_cases)
    assert result.confident_link is None or isinstance(
        result.confident_link, str
    )
    assert result.test_run_id is None or isinstance(result.test_run_id, str)

    for tr in result.test_results:
        assert isinstance(tr, TestResult)
        assert isinstance(tr.name, str) and tr.name != ""
        assert isinstance(tr.success, bool)
        assert tr.conversational is False

        # Single-turn results should have input/output
        assert tr.input is not None
        assert tr.actual_output is not None

        # Metrics data should be present
        assert isinstance(tr.metrics_data, list)
        assert len(tr.metrics_data) >= 1

    # Test JSON artifact schema
    json_records = save_dataset_as_json_and_load(
        dataset, directory=tmp_path, file_name="dataset"
    )
    assert isinstance(json_records, list)
    assert len(json_records) >= 1

    required_keys = {
        "input",
        "actual_output",
        "expected_output",
        "retrieval_context",
        "context",
        "name",
        "comments",
        "source_file",
        "tools_called",
        "expected_tools",
        "additional_metadata",
        "custom_column_key_values",
    }

    for rec in json_records:
        assert isinstance(rec, dict)
        assert required_keys.issubset(rec.keys())
        assert isinstance(rec["input"], str) and rec["input"] != ""

        # Optional fields can be None or their expected type
        assert rec["expected_output"] is None or isinstance(
            rec["expected_output"], str
        )
        assert rec["actual_output"] is None or isinstance(
            rec["actual_output"], str
        )
        assert rec["retrieval_context"] is None or isinstance(
            rec["retrieval_context"], list
        )
        assert rec["tools_called"] is None or isinstance(
            rec["tools_called"], list
        )
        assert rec["expected_tools"] is None or isinstance(
            rec["expected_tools"], list
        )


# ===========================================================================
# Checklist Item 2: Multi-turn E2E evaluation (doc-driven, offline)
# ===========================================================================


def test_docs_multi_turn_e2e_evaluation_result_shape(tmp_path):
    """
    Doc-driven regression test for multi-turn/conversational E2E evaluation.

    From the docs (Multi-Turn E2E Evals section):
      - Create ConversationalGolden with scenario, expected_outcome, user_description
      - Use ConversationSimulator to generate ConversationalTestCase objects
      - Call evaluate() with conversational_test_cases and conversational metrics

    Since ConversationSimulator requires a simulator_model (network), we manually
    construct ConversationalTestCase objects using a deterministic chatbot callback.

    Deterministic/offline: uses deterministic callback and metric (no network).
    """
    # Build multi-turn dataset (as shown in docs)
    dataset = build_multi_turn_dataset()
    assert len(dataset.goldens) >= 1

    # Manually build conversational test cases (offline alternative to ConversationSimulator)
    conversational_test_cases = build_conversational_test_cases_manually(
        dataset, max_turns=4
    )
    assert len(conversational_test_cases) == len(dataset.goldens)

    # Run evaluation with conversational metric
    result = evaluate(
        test_cases=conversational_test_cases,
        metrics=[DeterministicConversationalMetric()],
        hyperparameters={"model": "offline-stub", "system_prompt": "offline"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # Verify result is the expected type with correct structure
    assert isinstance(result, EvaluationResult)
    assert isinstance(result.test_results, list)
    assert len(result.test_results) == len(conversational_test_cases)
    assert result.confident_link is None or isinstance(
        result.confident_link, str
    )
    assert result.test_run_id is None or isinstance(result.test_run_id, str)

    # Verify test result structure for conversational test cases
    for tr in result.test_results:
        assert isinstance(tr, TestResult)
        assert isinstance(tr.name, str) and tr.name != ""
        assert isinstance(tr.success, bool)
        assert tr.conversational is True  # Multi-turn should be conversational

        # Metrics data should be present
        assert isinstance(tr.metrics_data, list)
        assert len(tr.metrics_data) >= 1


# ===========================================================================
# Checklist Item 3: JSON artifact schema for multi-turn dataset
# ===========================================================================


def test_docs_multi_turn_dataset_json_artifact_schema(tmp_path):
    """
    Doc-driven regression test for multi-turn dataset JSON export (Option A).

    From the docs:
        dataset = EvaluationDataset(goldens)
        dataset.save_as(file_type="json", directory="./example")

    Multi-turn JSON should contain:
      - scenario, turns, expected_outcome, user_description, context, name,
        comments, additional_metadata, custom_column_key_values
    """
    dataset = build_multi_turn_dataset()

    # Save and load JSON (Option A from docs)
    json_records = save_dataset_as_json_and_load(
        dataset, directory=tmp_path, file_name="multi_turn_dataset"
    )

    assert isinstance(json_records, list)
    assert len(json_records) >= 1

    # Multi-turn JSON schema (from dataset.py save_as implementation)
    required_keys = {
        "scenario",
        "turns",
        "expected_outcome",
        "user_description",
        "context",
        "name",
        "comments",
        "additional_metadata",
        "custom_column_key_values",
    }

    for rec in json_records:
        assert isinstance(rec, dict)
        assert required_keys.issubset(
            rec.keys()
        ), f"Missing keys: {required_keys - set(rec.keys())}"

        # Optional fields can be None or their expected type
        assert rec["scenario"] is None or isinstance(rec["scenario"], str)
        assert rec["turns"] is None or isinstance(rec["turns"], list)
        assert rec["expected_outcome"] is None or isinstance(
            rec["expected_outcome"], str
        )
        assert rec["user_description"] is None or isinstance(
            rec["user_description"], str
        )


# ===========================================================================
# Test: Single-turn CSV artifact schema (documented in save_as)
# ===========================================================================


def test_docs_single_turn_dataset_csv_artifact_schema(tmp_path):
    """
    Doc-driven regression test for single-turn dataset CSV export.

    From the docs:
        dataset = EvaluationDataset(goldens)
        dataset.save_as(file_type="csv", directory="./example")

    CSV should contain columns matching the Golden fields.
    """
    dataset = build_single_turn_dataset()

    csv_records = save_dataset_as_csv_and_load(
        dataset, directory=tmp_path, file_name="single_turn_dataset"
    )

    assert isinstance(csv_records, list)
    assert len(csv_records) >= 1

    # CSV should have "input" column at minimum
    for rec in csv_records:
        assert isinstance(rec, dict)
        assert "input" in rec.keys(), "CSV must have 'input' column"
        assert rec["input"] is not None and rec["input"] != ""


# ===========================================================================
# Test: Metric success/failure propagation
# ===========================================================================


def test_docs_evaluate_propagates_metric_failure():
    """
    Doc-driven regression test: verify that when a metric fails,
    the test result's success field is False.

    This ensures the documented behavior:
        "Apply metrics to your test cases and run evaluations"
    correctly propagates metric outcomes to test results.
    """
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicFailingMetric()],
        hyperparameters={"model": "offline-stub", "system_prompt": "offline"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)

    # All test results should be failures since the metric always fails
    for tr in result.test_results:
        assert (
            tr.success is False
        ), "Test result should be False when metric fails"
        assert len(tr.metrics_data) >= 1
        for md in tr.metrics_data:
            assert md.success is False


def test_docs_evaluate_propagates_metric_success():
    """
    Doc-driven regression test: verify that when a metric passes,
    the test result's success field is True.
    """
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters={"model": "offline-stub", "system_prompt": "offline"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)

    # All test results should be successes since the metric always passes
    for tr in result.test_results:
        assert (
            tr.success is True
        ), "Test result should be True when metric passes"
        assert len(tr.metrics_data) >= 1
        for md in tr.metrics_data:
            assert md.success is True


# ===========================================================================
# Test: Multiple metrics in single evaluation
# ===========================================================================


def test_docs_evaluate_with_multiple_metrics():
    """
    Doc-driven regression test for evaluating with multiple metrics.

    From the docs:
        "metrics: a list of metrics of type BaseMetric"

    When multiple metrics are provided, all should be evaluated and
    their results included in metrics_data.
    """
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    metrics = [
        DeterministicPassingMetric(),
        DeterministicContainsExpectedOutputMetric(),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={"model": "offline-stub", "system_prompt": "offline"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)

    for tr in result.test_results:
        # Should have results from all metrics
        assert len(tr.metrics_data) == len(
            metrics
        ), f"Expected {len(metrics)} metric results, got {len(tr.metrics_data)}"

        # Verify each metric result has a non-empty name
        for md in tr.metrics_data:
            assert isinstance(md.name, str) and md.name != ""


# ===========================================================================
# Test: Dataset loading from JSON file (documented flow)
# ===========================================================================


def test_docs_add_goldens_from_json_file_flow(tmp_path):
    """
    Doc-driven regression test for the JSON loading flow.

    From the docs:
        dataset = EvaluationDataset()
        dataset.add_goldens_from_json_file(file_path="example.json", input_key_name="query")

    This tests the flow: write minimal JSON -> load -> evaluate.
    """
    # Write a minimal JSON file directly (not via save_as, to avoid round-trip issues)
    json_path = tmp_path / "test_goldens.json"
    goldens_data = [
        {
            "input": "What is your name?",
            "expected_output": "My name is DeepEval.",
        },
        {"input": "Choose a number between 1 to 100", "expected_output": "42"},
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(goldens_data, f)

    # Load into a new dataset (as shown in docs)
    loaded_dataset = EvaluationDataset()
    loaded_dataset.add_goldens_from_json_file(
        file_path=str(json_path),
        input_key_name="input",
    )

    # Verify goldens were loaded correctly
    assert len(loaded_dataset.goldens) == len(goldens_data)

    for orig, loaded in zip(goldens_data, loaded_dataset.goldens):
        assert loaded.input == orig["input"]
        assert loaded.expected_output == orig["expected_output"]

    # Now evaluate using the loaded dataset (completing the documented flow)
    test_cases = build_llm_test_cases_from_goldens(loaded_dataset)

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicContainsExpectedOutputMetric()],
        hyperparameters={"model": "offline-stub", "system_prompt": "offline"},
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)

    # All should pass since our deterministic LLM app returns expected outputs
    for tr in result.test_results:
        assert (
            tr.success is True
        ), f"Test case '{tr.input}' failed unexpectedly after JSON load"


# ===========================================================================
# Test: Hyperparameters are accepted by evaluate() (documented optional param)
# ===========================================================================


def test_docs_evaluate_accepts_hyperparameters():
    """
    Doc-driven regression test for hyperparameters parameter.

    From the docs:
        "[Optional] hyperparameters: a dict of type dict[str, Union[str, int, float]].
        You can log any arbitrary hyperparameter associated with this test run..."

    This verifies evaluate() accepts hyperparameters without error.
    """
    dataset = build_single_turn_dataset()
    test_cases = build_llm_test_cases_from_goldens(dataset)

    # Test with various hyperparameter types as documented
    hyperparameters = {
        "model": "gpt-4.1",
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    result = evaluate(
        test_cases=test_cases,
        metrics=[DeterministicPassingMetric()],
        hyperparameters=hyperparameters,
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )

    # Should complete without error
    assert isinstance(result, EvaluationResult)
    assert len(result.test_results) == len(test_cases)


# ===========================================================================
# Checklist Item 4: CLI smoke test (networked, requires OPENAI_API_KEY)
# ===========================================================================


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping networked CLI smoke test",
)
def test_docs_cli_smoke_test_networked(tmp_path):
    """
    CLI smoke test for `deepeval test run`.

    From the docs (evaluation-end-to-end-llm-evals):
        deepeval test run test_example.py

    This test requires OPENAI_API_KEY to be set and will be skipped otherwise.
    It creates a minimal test file and runs `poetry run deepeval test run` on it.
    """

    # Create a minimal test file that uses DeepEval CLI
    # Note: file must start with "test_" prefix for deepeval CLI
    test_file = tmp_path / "test_cli_smoke.py"
    test_file.write_text(
        '''
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_cli_smoke():
    """Minimal test case for CLI smoke test."""
    test_case = LLMTestCase(
        input="What is 2+2?",
        actual_output="4",
    )
    assert_test(test_case, metrics=[AnswerRelevancyMetric(threshold=0.5)])
'''
    )

    # Run the CLI via subprocess through Poetry
    proc = subprocess.run(
        ["poetry", "run", "deepeval", "test", "run", str(test_file)],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout for network calls
    )

    # Assert CLI completed successfully
    assert proc.returncode == 0, (
        f"CLI smoke test failed with return code {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )


# ===========================================================================
# ErrorConfig.skip_on_missing_params tests
# Docs: "skip_on_missing_params=True skips all metric executions for test
#        cases with missing parameters"
# ===========================================================================


def test_docs_error_config_skip_on_missing_params_false_raises():
    """
    Doc-driven test for ErrorConfig.skip_on_missing_params=False (default).

    From docs (evaluation-flags-and-configs.mdx):
        "skip_on_missing_params: a boolean which when set to True, skips all
         metric executions for test cases with missing parameters. Defaulted
         to False."

    When False (default), a metric requiring a missing param should raise
    MissingTestCaseParamsError.
    """
    # Create a test case WITHOUT retrieval_context
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


def test_docs_error_config_skip_on_missing_params_true_skips_metric():
    """
    Doc-driven test for ErrorConfig.skip_on_missing_params=True.

    From docs:
        "skip_on_missing_params=True skips all metric executions for test
         cases with missing parameters"

    When True, metrics requiring missing params should be skipped and the
    result should NOT contain metrics_data for that metric.
    """
    # Create a test case WITHOUT retrieval_context
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


def test_docs_error_config_skip_on_missing_params_with_valid_test_case():
    """
    Doc-driven test: when test case HAS all required params,
    skip_on_missing_params=True should NOT skip the metric.
    """
    # Create a test case WITH retrieval_context
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


def test_docs_error_config_skip_on_missing_params_takes_precedence():
    """
    Doc-driven test for precedence behavior.

    From docs:
        "If both skip_on_missing_params and ignore_errors are set to True,
         skip_on_missing_params takes precedence. This means that if a metric
         is missing required test case parameters, it will be skipped (and the
         result will be missing) rather than appearing as an ignored error in
         the final test run."

    When both are True and params are missing, the metric should be skipped
    (not appear in metrics_data) rather than show as an ignored error.
    """
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


# ===========================================================================
# ErrorConfig.ignore_errors tests
# Docs: "ignore_errors=True ignores all exceptions raised during metrics
#        execution for each test case"
# ===========================================================================


def test_docs_error_config_ignore_errors_false_raises():
    """
    Doc-driven test for ErrorConfig.ignore_errors=False (default).

    From docs:
        "ignore_errors: a boolean which when set to True, ignores all
         exceptions raised during metrics execution for each test case.
         Defaulted to False."

    When False (default), a metric that raises should propagate the exception.
    """
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


def test_docs_error_config_ignore_errors_true_captures_error():
    """
    Doc-driven test for ErrorConfig.ignore_errors=True.

    From docs:
        "ignore_errors=True ignores all exceptions raised during metrics
         execution for each test case"

    When True, exceptions should be captured and the evaluation should
    complete. The metric should show as failed with an error message.
    """
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


def test_docs_error_config_ignore_errors_with_mixed_metrics():
    """
    Doc-driven test: ignore_errors should only affect metrics that raise,
    not metrics that succeed.
    """
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


# ===========================================================================
# AsyncConfig tests
# Docs: "run_async=True enables concurrent evaluation of test cases AND metrics"
# ===========================================================================


def test_docs_async_config_sync_vs_async_equivalent_results():
    """
    Doc-driven test for AsyncConfig.run_async variations.

    From docs:
        "run_async: a boolean which when set to True, enables concurrent
         evaluation of test cases AND metrics. Defaulted to True."

    Both sync and async execution should produce equivalent results for
    deterministic metrics.
    """
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


def test_docs_async_config_max_concurrent_validation():
    """
    Doc-driven test for AsyncConfig.max_concurrent validation.

    From docs:
        "max_concurrent: an integer that determines the maximum number of
         test cases that can be ran in parallel at any point in time."

    This verifies max_concurrent must be at least 1.
    """
    with pytest.raises(ValueError, match="max_concurrent"):
        AsyncConfig(max_concurrent=0)


def test_docs_async_config_throttle_value_validation():
    """
    Doc-driven test for AsyncConfig.throttle_value validation.

    From docs:
        "throttle_value: an integer that determines how long (in seconds)
         to throttle the evaluation of each test case."

    This verifies throttle_value must be non-negative.
    """
    with pytest.raises(ValueError, match="throttle_value"):
        AsyncConfig(throttle_value=-1)


# ===========================================================================
# CacheConfig tests
# Docs: "write_cache=True writes test run results to DISK"
# ===========================================================================


def test_docs_cache_config_disabled_caching_works():
    """
    Doc-driven test for CacheConfig with caching disabled.

    From docs:
        "write_cache: a boolean which when set to True, uses writes test
         run results to DISK. Defaulted to True."
        "The write_cache parameter writes to disk and so you should disable
         it if that is causing any errors in your environment."

    This verifies evaluation works correctly with caching completely disabled.
    """
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


# ===========================================================================
# DisplayConfig tests
# Docs: "display: a str of either 'all', 'failing' or 'passing'"
# ===========================================================================


def test_docs_display_config_display_all():
    """
    Doc-driven test for DisplayConfig.display_option="all".

    From docs:
        "display: a str of either 'all', 'failing' or 'passing', which allows
         you to selectively decide which type of test cases to display as the
         final result. Defaulted to 'all'."

    Verifies that display="all" does not crash and returns expected results.
    Note: display_option only affects printing, not underlying evaluation.
    """
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


def test_docs_display_config_display_failing():
    """
    Doc-driven test for DisplayConfig.display_option="failing".

    Verifies that display="failing" does not crash and returns expected results.
    Note: display_option only affects printing, not underlying evaluation.
    """
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


def test_docs_display_config_display_passing():
    """
    Doc-driven test for DisplayConfig.display_option="passing".

    Verifies that display="passing" does not crash and returns expected results.
    Note: display_option only affects printing, not underlying evaluation.
    """
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


def test_docs_display_config_does_not_affect_evaluation_results():
    """
    Doc-driven test: DisplayConfig options should NOT affect evaluation results,
    only the display/printing behavior.

    Run the same evaluation with different display options and verify
    identical evaluation outcomes.
    """
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
