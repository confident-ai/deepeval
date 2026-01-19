import json
import os
import subprocess
import pytest

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult, TestResult

from .helpers import (
    DeterministicContainsExpectedOutputMetric,
    DeterministicConversationalMetric,
    DeterministicFailingMetric,
    DeterministicPassingMetric,
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
