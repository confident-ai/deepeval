from pathlib import Path

import pytest

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("file_type", ["md", "html"])
def test_report_export_does_not_depend_on_print_results(
    tmp_path: Path, file_type: str, run_async: bool
):
    test_case = LLMTestCase(
        input="Capital of France?",
        actual_output="Paris",
        expected_output="Paris",
    )

    evaluate(
        test_cases=[test_case],
        metrics=[ExactMatchMetric()],
        async_config=AsyncConfig(run_async=run_async),
        display_config=DisplayConfig(
            print_results=False,
            show_indicator=False,
            file_type=file_type,
            file_output_dir=str(tmp_path),
        ),
    )

    assert len(list(tmp_path.glob(f"*.{file_type}"))) == 1
