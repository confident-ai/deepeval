import csv
import json

import pytest

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase


@pytest.mark.parametrize(
    "metadata",
    [
        {"enabled": True},
        {"enabled": False},
        {"missing": None},
        {"nested": {"values": [True, False, None]}, "label": "true/null"},
        {},
    ],
)
def test_test_case_csv_metadata_round_trip(tmp_path, metadata):
    dataset = EvaluationDataset()
    dataset.add_test_case(
        LLMTestCase(
            input="question",
            actual_output="answer",
            metadata=metadata,
        )
    )
    path = dataset.save_as("csv", str(tmp_path), include_test_cases=True)

    restored = EvaluationDataset()
    restored.add_test_cases_from_csv_file(path, "input", "actual_output")

    assert restored.test_cases[0].metadata == metadata


@pytest.mark.parametrize(
    "cell, expected",
    [
        (
            json.dumps({"enabled": True, "missing": None}),
            {"enabled": True, "missing": None},
        ),
        (
            "{'enabled': True, 'missing': None}",
            {"enabled": True, "missing": None},
        ),
        ("{'label': 'legacy', 'count': 2}", {"label": "legacy", "count": 2}),
        ("", None),
    ],
)
def test_test_case_csv_metadata_formats(tmp_path, cell, expected):
    path = tmp_path / "test_cases.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input", "actual_output", "metadata"])
        writer.writerow(["question", "answer", cell])

    dataset = EvaluationDataset()
    dataset.add_test_cases_from_csv_file(
        str(path),
        "input",
        "actual_output",
        additional_metadata_col_name="metadata",
    )

    assert dataset.test_cases[0].metadata == expected


def test_test_case_csv_missing_metadata_column(tmp_path):
    path = tmp_path / "test_cases.csv"
    path.write_text("input,actual_output\nquestion,answer\n")
    dataset = EvaluationDataset()

    dataset.add_test_cases_from_csv_file(str(path), "input", "actual_output")

    assert dataset.test_cases[0].metadata is None


def test_test_case_csv_invalid_metadata_still_raises(tmp_path):
    path = tmp_path / "test_cases.csv"
    path.write_text(
        "input,actual_output,additional_metadata\nquestion,answer,not-a-mapping\n"
    )
    dataset = EvaluationDataset()

    with pytest.raises((ValueError, SyntaxError)):
        dataset.add_test_cases_from_csv_file(
            str(path), "input", "actual_output"
        )
