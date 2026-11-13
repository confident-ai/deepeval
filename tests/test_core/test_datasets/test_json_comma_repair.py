import csv
import json

import pytest

from deepeval.dataset.utils import trimAndLoadJson
from deepeval.dataset import EvaluationDataset


@pytest.mark.parametrize(
    "value",
    [
        "items A, B, ] then stopped",
        "the set is {x, y, } here",
        'quoted "value, ]" remains intact',
        "a backslash \\ followed by , }",
        "line one\nline two, ]",
        "Unicode: café, ]",
    ],
)
def test_trailing_comma_repair_preserves_string_values(value):
    raw = json.dumps({"text": value, "items": [1, 2]})
    # A structural trailing comma requires repair, but the text is untouched.
    malformed = raw[:-1] + ",}"
    assert trimAndLoadJson(malformed) == json.loads(raw)


def test_repairs_nested_trailing_commas_without_rewriting_strings():
    raw = '{"items": [{"text": "keep, ] and , }",},],}'
    assert trimAndLoadJson(raw) == {"items": [{"text": "keep, ] and , }"}]}


@pytest.mark.parametrize(
    "raw", ['{"text": "unterminated,}', '{"a":,}', "[1,,]"]
)
def test_other_malformed_json_still_raises(raw):
    with pytest.raises(ValueError):
        trimAndLoadJson(raw)


def test_csv_tool_arguments_survive_trailing_comma_repair(tmp_path):
    path = tmp_path / "goldens.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input", "tools_called"])
        writer.writeheader()
        writer.writerow(
            {
                "input": "Search for this exact text",
                "tools_called": (
                    '[{"name": "search", '
                    '"input_parameters": {"query": "literal, ]"},},]'
                ),
            }
        )

    dataset = EvaluationDataset()
    dataset.add_goldens_from_csv_file(str(path))
    assert dataset.goldens[0].tools_called[0].input_parameters == {
        "query": "literal, ]"
    }
