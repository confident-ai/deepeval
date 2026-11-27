import json

import pytest

from deepeval.dataset import ConversationalGolden, EvaluationDataset


@pytest.mark.parametrize("file_type", ["json", "jsonl"])
@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, "default_lookup"),
        ({"expected_tools_key_name": "tool_expectations"}, "custom_lookup"),
        ({"expected_tools_key_name": "missing"}, None),
        ({"expected_tools_key_name": None}, None),
    ],
)
def test_conversational_goldens_respect_expected_tools_key(
    tmp_path, file_type, options, expected
):
    record = {
        "scenario": "Find the weather.",
        "expected_tools": [{"name": "default_lookup"}],
        "tool_expectations": [{"name": "custom_lookup"}],
    }
    data = [record] if file_type == "json" else record
    path = tmp_path / f"goldens.{file_type}"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    dataset = EvaluationDataset()
    load = getattr(dataset, f"add_goldens_from_{file_type}_file")
    load(str(path), **options)

    (golden,) = dataset.goldens
    assert isinstance(golden, ConversationalGolden)
    if expected is None:
        assert golden.expected_tools is None
    else:
        (tool,) = golden.expected_tools
        assert tool.name == expected
