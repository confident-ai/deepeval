import pytest
import os
import tempfile
import json
import csv
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.test_case import (
    Turn,
    LLMTestCase,
    ConversationalTestCase,
)


class TestSaveAndLoad:
    def test_dataset_save_load_goldens(self):
        """Load Goldens from both CSV and JSON and check their count and a sample field."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        json_path = os.path.join(current_dir, "goldens.json")
        csv_path = os.path.join(current_dir, "goldens.csv")

        dataset_json = EvaluationDataset()
        dataset_csv = EvaluationDataset()
        dataset_json.add_goldens_from_json_file(file_path=json_path)
        dataset_csv.add_goldens_from_csv_file(file_path=csv_path)

        assert len(dataset_json.goldens) == 15
        assert len(dataset_csv.goldens) == 15
        assert all(golden.input is not None for golden in dataset_json.goldens)
        assert all(golden.input is not None for golden in dataset_csv.goldens)

    def test_dataset_save_load_conversational_goldens(self):
        """Load ConversationalGoldens from both CSV and JSON and check their count and a sample field."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        json_path = os.path.join(current_dir, "convo_goldens.json")
        csv_path = os.path.join(current_dir, "convo_goldens.csv")

        dataset_json = EvaluationDataset()
        dataset_csv = EvaluationDataset()
        dataset_json.add_goldens_from_json_file(file_path=json_path)
        dataset_csv.add_goldens_from_csv_file(file_path=csv_path)

        assert len(dataset_json.goldens) == 15
        assert len(dataset_csv.goldens) == 15
        assert all(
            golden.scenario is not None for golden in dataset_json.goldens
        )
        assert all(
            golden.scenario is not None for golden in dataset_csv.goldens
        )

    def test_save_as_creates_valid_json_and_csv(self):
        """Test saving goldens as JSON and CSV to temp files."""
        goldens = [
            Golden(
                input="Test input",
                expected_output="Test output",
                actual_output="Test output",
                retrieval_context=["context1"],
                context=["test"],
                source_file="source.txt",
            )
        ]
        dataset = EvaluationDataset(goldens)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = dataset.save_as(
                "json", directory=tmpdir, file_name="goldens_test"
            )
            assert os.path.exists(json_path)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, list)
                assert data[0]["input"] == "Test input"

            csv_path = dataset.save_as(
                "csv", directory=tmpdir, file_name="goldens_test_csv"
            )
            assert os.path.exists(csv_path)
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                header = rows[0]
                data_row = rows[1]
                assert header[0] == "input"
                assert data_row[0] == "Test input"

    def test_save_as_conversational_goldens_creates_valid_json_and_csv(self):
        """Test saving ConversationalGoldens as JSON and CSV to temp files."""
        convo_goldens = [
            ConversationalGolden(
                scenario="Book a flight to Tokyo",
                expected_outcome="User gets flight options",
                user_description="User is trying to find flights",
                context=["Flights", "Travel"],
                turns=[
                    Turn(role="user", content="Find me a flight to Tokyo"),
                    Turn(
                        role="assistant",
                        content="Here are some flight options to Tokyo",
                    ),
                ],
            )
        ]

        dataset = EvaluationDataset(convo_goldens)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = dataset.save_as(
                "json", directory=tmpdir, file_name="test_convo_json"
            )
            assert os.path.exists(json_path)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, list)
                assert data[0]["scenario"] == "Book a flight to Tokyo"
                assert "turns" in data[0]
                assert isinstance(
                    data[0]["turns"], str
                )  # Turns are formatted into | seperated values

            csv_path = dataset.save_as(
                "csv", directory=tmpdir, file_name="test_convo_csv"
            )
            assert os.path.exists(csv_path)
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                assert len(rows) >= 2
                assert "Book a flight to Tokyo" in rows[1]

    def test_save_as_empty_dataset_raises_error(self):
        """Test that calling save_as on an empty dataset raises a ValueError."""
        dataset = EvaluationDataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No goldens found"):
                dataset.save_as("json", directory=tmpdir)

    def test_save_as_includes_test_cases(self):
        """Check that test cases get included when include_test_cases=True."""
        test_case = LLMTestCase(
            input="input case",
            actual_output="actual",
            context=["test"],
            retrieval_context=["ctx"],
        )
        dataset = EvaluationDataset()
        dataset.add_test_case(test_case)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = dataset.save_as(
                "json", directory=tmpdir, include_test_cases=True
            )
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert any(item["input"] == "input case" for item in data)

    def test_save_as_includes_convo_test_cases(self):
        """Check that convo test cases get included when include_test_cases=True."""
        test_case = ConversationalTestCase(
            scenario="test case scenario",
            turns=[
                Turn(role="user", content="user content"),
                Turn(role="assistant", content="assistant content"),
            ],
        )
        dataset = EvaluationDataset()
        dataset._multi_turn = True
        dataset.add_test_case(test_case)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = dataset.save_as(
                "json", directory=tmpdir, include_test_cases=True
            )
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert any(
                    item["scenario"] == "test case scenario" for item in data
                )
