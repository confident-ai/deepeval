import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import DisplayConfig


def test_read_only_evaluation(monkeypatch, capsys, tmp_path):
    # Set the read-only environment variable
    monkeypatch.setenv("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")

    # Create a dummy test case and metric
    test_case = LLMTestCase(input="input", actual_output="actual")
    metric = AnswerRelevancyMetric()

    # Create a display config to control output
    display_config = DisplayConfig(
        print_results=False, file_output_dir=str(tmp_path)
    )

    # Attempt to evaluate and write the results
    evaluate([test_case], [metric], display_config=display_config)

    # Check that no files were created in the directory
    assert not any(tmp_path.iterdir())

    # Check that the warning was printed
    captured = capsys.readouterr()
    assert "Warning: Skipping write due to DEEPEVAL_FILE_SYSTEM=READ_ONLY" in captured.out 