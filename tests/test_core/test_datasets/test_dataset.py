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
    ToolCall,
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
        assert all(golden.name is not None for golden in dataset_json.goldens)
        assert all(
            golden.comments is not None for golden in dataset_csv.goldens
        )

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
        assert all(golden.name is not None for golden in dataset_json.goldens)
        assert all(
            golden.comments is not None for golden in dataset_csv.goldens
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
                name="Name",
                comments="Comment",
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
                name="Name",
                comments="Comment",
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
                # Turns are now structured arrays, not lossy strings
                assert isinstance(data[0]["turns"], list)
                assert data[0]["turns"][0]["role"] == "user"
                assert (
                    data[0]["turns"][0]["content"]
                    == "Find me a flight to Tokyo"
                )

            csv_path = dataset.save_as(
                "csv", directory=tmpdir, file_name="test_convo_csv"
            )
            assert os.path.exists(csv_path)
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                assert len(rows) >= 2
                assert "Book a flight to Tokyo" in rows[1]

    def test_save_as_includes_extra_single_turn_fields(self):
        """Single-turn JSON/CSV/JSONL include tools/metadata/custom columns."""
        goldens = [
            Golden(
                input="Ask",
                expected_output="Ans",
                actual_output="Ans",
                retrieval_context=["rctx"],
                context=["ctx"],
                source_file="src.txt",
                name="n",
                comments="c",
                tools_called=[
                    ToolCall(
                        name="search",
                        input_parameters={"q": "foo"},
                        output={"ok": True},
                    )
                ],
                expected_tools=[ToolCall(name="finalize")],
                additional_metadata={"k": "v"},
                custom_column_key_values={"col": "val"},
            )
        ]
        dataset = EvaluationDataset(goldens)

        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON
            json_path = dataset.save_as(
                "json", directory=tmpdir, file_name="single_json"
            )
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                row = data[0]
                assert (
                    isinstance(row["tools_called"], list)
                    and row["tools_called"][0]["name"] == "search"
                )
                assert (
                    isinstance(row["expected_tools"], list)
                    and row["expected_tools"][0]["name"] == "finalize"
                )
                assert row["additional_metadata"]["k"] == "v"
                assert row["custom_column_key_values"]["col"] == "val"

            # JSONL
            jsonl_path = dataset.save_as(
                "jsonl", directory=tmpdir, file_name="single_jsonl"
            )
            with open(jsonl_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                row = json.loads(line)
                assert (
                    isinstance(row["tools_called"], list)
                    and row["tools_called"][0]["name"] == "search"
                )
                assert (
                    isinstance(row["expected_tools"], list)
                    and row["expected_tools"][0]["name"] == "finalize"
                )
                assert row["additional_metadata"]["k"] == "v"
                assert row["custom_column_key_values"]["col"] == "val"

            # CSV
            csv_path = dataset.save_as(
                "csv", directory=tmpdir, file_name="single_csv"
            )
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                header = rows[0]
                vals = rows[1]
                assert "tools_called" in header and "expected_tools" in header
                # Find column indices
                tools_idx = header.index("tools_called")
                expected_idx = header.index("expected_tools")
                meta_idx = header.index("additional_metadata")
                custom_idx = header.index("custom_column_key_values")
                # Validate JSON-encoded cells are present
                assert vals[tools_idx]
                assert vals[expected_idx]
                # Parse back to ensure valid JSON
                tools_arr = json.loads(vals[tools_idx])
                assert tools_arr[0]["name"] == "search"
                expected_arr = json.loads(vals[expected_idx])
                assert expected_arr[0]["name"] == "finalize"
                if vals[meta_idx]:
                    meta_obj = json.loads(vals[meta_idx])
                    assert meta_obj["k"] == "v"
                if vals[custom_idx]:
                    custom_obj = json.loads(vals[custom_idx])
                    assert custom_obj["col"] == "val"

    def test_save_as_includes_turn_fields_in_multi_turn_json_and_jsonl(self):
        """Multi-turn JSON/JSONL include full turn fields (user_id, tools)."""
        convo = [
            ConversationalGolden(
                scenario="s",
                expected_outcome="eo",
                user_description="ud",
                context=["ctx"],
                turns=[
                    Turn(
                        role="user",
                        content="hi",
                        user_id="u1",
                        retrieval_context=["r"],
                        tools_called=[ToolCall(name="t")],
                        additional_metadata={"mk": "mv"},
                    ),
                ],
                name="n",
                comments="c",
                additional_metadata={"gk": "gv"},
                custom_column_key_values={"col": "val"},
            )
        ]
        dataset = EvaluationDataset(convo)

        with tempfile.TemporaryDirectory() as tmpdir:
            p_json = dataset.save_as(
                "json", directory=tmpdir, file_name="convo_json"
            )
            with open(p_json, "r", encoding="utf-8") as f:
                data = json.load(f)[0]
                turns = data["turns"]
                assert isinstance(turns, list) and turns[0]["user_id"] == "u1"
                assert isinstance(turns[0]["tools_called"], list)
                assert data["additional_metadata"]["gk"] == "gv"
                assert data["custom_column_key_values"]["col"] == "val"

            p_jsonl = dataset.save_as(
                "jsonl", directory=tmpdir, file_name="convo_jsonl"
            )
            with open(p_jsonl, "r", encoding="utf-8") as f:
                rec = json.loads(f.readline())
                turns = rec["turns"]
                assert isinstance(turns, list) and turns[0]["user_id"] == "u1"
                assert isinstance(turns[0]["tools_called"], list)

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
            name="Name",
            comments="Comment",
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
            name="Name",
            comments="Comment",
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
