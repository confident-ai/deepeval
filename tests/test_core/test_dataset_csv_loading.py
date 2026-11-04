"""CSV loading and round-tripping for EvaluationDataset.

These live outside tests/test_core/test_datasets/ because that directory is
only run in the job that has API keys; nothing here needs one.
"""

import csv
import os
import tempfile

from deepeval.dataset import EvaluationDataset, Golden


def test_csv_loaders_keep_na_like_text():
    """pandas' default NA tokens ("N/A", "None", "null", ...) are answer
    text in an eval set, not missing values, and a numeric-looking column
    is text as well. Only an empty cell is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "na.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "input",
                    "actual_output",
                    "expected_output",
                    "token_cost",
                    "input_token_count",
                ]
            )
            writer.writerow(["q1", "N/A", "N/A", "0.5", "12"])
            writer.writerow(["q2", "None", "null", "1", "3"])
            writer.writerow(["q3", "Paris", "", "", ""])

        test_cases = EvaluationDataset()
        test_cases.add_test_cases_from_csv_file(path, "input", "actual_output")
        assert [t.actual_output for t in test_cases.test_cases] == [
            "N/A",
            "None",
            "Paris",
        ]
        assert [t.expected_output for t in test_cases.test_cases] == [
            "N/A",
            "null",
            None,
        ]

        goldens = EvaluationDataset()
        goldens.add_goldens_from_csv_file(path)
        assert [g.actual_output for g in goldens.goldens] == [
            "N/A",
            "None",
            "Paris",
        ]
        # Golden still coerces its numeric fields from the text cells
        assert [g.token_cost for g in goldens.goldens] == [0.5, 1.0, None]
        assert [g.input_token_count for g in goldens.goldens] == [
            12,
            3,
            None,
        ]


def test_save_as_csv_round_trips_na_like_and_numeric_strings():
    """A CSV written by save_as has to load back unchanged. A column whose
    cells all look numeric used to come back as numbers, which Golden
    rejects, and "N/A" came back as None."""
    goldens = [
        Golden(input="unanswerable", expected_output="N/A", name="1"),
        Golden(input="6*7?", expected_output="42", name="2"),
        Golden(input="pad 7 to 3 digits", expected_output="007", name="3"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = EvaluationDataset(goldens).save_as(
            "csv", tmpdir, file_name="rt_na"
        )

        reloaded = EvaluationDataset()
        reloaded.add_goldens_from_csv_file(path)
        assert [g.expected_output for g in reloaded.goldens] == [
            "N/A",
            "42",
            "007",
        ]
        assert [g.name for g in reloaded.goldens] == ["1", "2", "3"]
        # an unwritten cell is still missing
        assert all(g.actual_output is None for g in reloaded.goldens)
