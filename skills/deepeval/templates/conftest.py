"""Shared pytest fixtures for eval suites.

Keep dataset loading explicit in each test file:

    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(file_path=DATASET_PATH)

Use `add_goldens_from_csv_file`, `add_goldens_from_jsonl_file`, or
`dataset.pull(alias=...)` instead when the dataset source requires it.
"""
