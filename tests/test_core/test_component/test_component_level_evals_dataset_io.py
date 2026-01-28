import csv
import json
import pytest
import os
from deepeval.dataset import EvaluationDataset, Golden

pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None
    or not os.getenv("OPENAI_API_KEY").strip(),
    reason="OPENAI_API_KEY is not set",
)


def test_dataset_add_goldens_from_json_file_loads_inputs(tmp_path):
    data = [
        {"prompt": "Ping?"},
        {"prompt": "Pong?"},
        {"prompt": "Hello!"},
    ]
    p = tmp_path / "goldens.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(
        file_path=str(p),
        input_key_name="prompt",
    )

    seen = [g.input for g in dataset.evals_iterator()]
    assert seen == ["Ping?", "Pong?", "Hello!"]


def test_dataset_add_goldens_from_csv_file_loads_inputs(tmp_path):
    p = tmp_path / "goldens.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "other"])
        w.writeheader()
        w.writerow({"prompt": "Ping?", "other": "x"})
        w.writerow({"prompt": "Pong?", "other": "y"})

    dataset = EvaluationDataset()
    dataset.add_goldens_from_csv_file(
        file_path=str(p),
        input_col_name="prompt",
    )

    # The loader should have produced single-turn Goldens
    assert [g.input for g in dataset.goldens] == ["Ping?", "Pong?"]
    assert all(isinstance(g, Golden) for g in dataset.goldens)

    seen = [g.input for g in dataset.evals_iterator()]
    assert seen == ["Ping?", "Pong?"]
