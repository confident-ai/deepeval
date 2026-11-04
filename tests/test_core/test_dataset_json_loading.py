import json

import pytest

from deepeval.dataset import EvaluationDataset


@pytest.mark.parametrize(
    "metadata_key", ["labels", "metadata", "additional_metadata"]
)
def test_json_import_preserves_selected_metadata(tmp_path, metadata_key):
    file_path = tmp_path / "test_cases.json"
    file_path.write_text(
        json.dumps(
            [
                {
                    "input": "How do I get a refund?",
                    "actual_output": "Contact customer support.",
                    "unselected_metadata": {"ignore": True},
                    metadata_key: {
                        "language": "zh",
                        "domain": "退款",
                        "attributes": {"critical": True, "note": None},
                        "tags": ["售后", "中文"],
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    dataset = EvaluationDataset()

    dataset.add_test_cases_from_json_file(
        file_path=str(file_path),
        input_key_name="input",
        actual_output_key_name="actual_output",
        addtional_metadata_key_name=metadata_key,
    )

    assert len(dataset.test_cases) == 1
    assert dataset.test_cases[0].metadata == {
        "language": "zh",
        "domain": "退款",
        "attributes": {"critical": True, "note": None},
        "tags": ["售后", "中文"],
    }


def test_json_import_keeps_metadata_aligned_with_test_cases(tmp_path):
    file_path = tmp_path / "test_cases.json"
    file_path.write_text(
        json.dumps(
            [
                {"input": "q1", "actual_output": "a1", "labels": {"row": 1}},
                {"input": "q2", "actual_output": "a2", "labels": {}},
                {"input": "q3", "actual_output": "a3", "labels": None},
                {"input": "q4", "actual_output": "a4"},
                {"input": "q5", "actual_output": "a5", "labels": {"row": 5}},
            ]
        ),
        encoding="utf-8",
    )
    dataset = EvaluationDataset()

    dataset.add_test_cases_from_json_file(
        file_path=str(file_path),
        input_key_name="input",
        actual_output_key_name="actual_output",
        addtional_metadata_key_name="labels",
    )

    assert [
        (case.input, case.actual_output, case.metadata)
        for case in dataset.test_cases
    ] == [
        ("q1", "a1", {"row": 1}),
        ("q2", "a2", {}),
        ("q3", "a3", None),
        ("q4", "a4", None),
        ("q5", "a5", {"row": 5}),
    ]


def test_json_import_leaves_metadata_unset_without_key_name(tmp_path):
    file_path = tmp_path / "test_cases.json"
    file_path.write_text(
        json.dumps(
            [
                {
                    "input": "q1",
                    "actual_output": "a1",
                    "labels": {"domain": "support"},
                    "metadata": {"domain": "support"},
                    "additional_metadata": {"domain": "support"},
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset = EvaluationDataset()

    dataset.add_test_cases_from_json_file(
        file_path=str(file_path),
        input_key_name="input",
        actual_output_key_name="actual_output",
    )

    assert len(dataset.test_cases) == 1
    assert dataset.test_cases[0].metadata is None
