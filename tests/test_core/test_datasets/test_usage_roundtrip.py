import csv
import json
from pathlib import Path

import pytest

from deepeval.dataset import EvaluationDataset, Golden


@pytest.mark.parametrize("file_type", ["json", "jsonl", "csv"])
@pytest.mark.parametrize(
    "usage",
    [
        {
            "token_cost": 0.025,
            "input_token_count": 37,
            "output_token_count": 11,
        },
        {"token_cost": 0.0, "input_token_count": 0, "output_token_count": 0},
        {
            "token_cost": None,
            "input_token_count": None,
            "output_token_count": None,
        },
    ],
)
def test_save_as_preserves_golden_usage(tmp_path, file_type, usage):
    golden = Golden(input="usage — 测试", **usage)
    dataset = EvaluationDataset(goldens=[golden])
    original = golden.model_dump()

    saved_path = Path(
        dataset.save_as(file_type, directory=str(tmp_path), file_name="usage")
    )

    with saved_path.open(encoding="utf-8", newline="") as file:
        if file_type == "csv":
            row = next(csv.DictReader(file))
            expected = {
                key: "" if value is None else str(value)
                for key, value in usage.items()
            }
        else:
            row = (
                json.load(file)[0]
                if file_type == "json"
                else json.loads(file.readline())
            )
            expected = usage
    assert {key: row[key] for key in usage} == expected

    loaded = EvaluationDataset()
    if file_type == "csv":
        loaded.add_goldens_from_csv_file(str(saved_path))
    elif file_type == "json":
        loaded.add_goldens_from_json_file(str(saved_path))
    else:
        loaded.add_goldens_from_jsonl_file(str(saved_path))

    assert loaded.goldens[0].input == golden.input
    assert {key: getattr(loaded.goldens[0], key) for key in usage} == usage
    assert golden.model_dump() == original
