# Datasets

Use documented `EvaluationDataset` APIs directly. Do not invent wrapper helpers
for dataset loading in templates.

If the user does not have a dataset yet, read `synthetic-data.md` and generate
one with `deepeval generate` before creating the pytest eval file.

If the user has a dataset, check its size before accepting it as sufficient.
Fewer than 10 goldens is very likely too small. A useful first eval dataset is
usually 50-100 goldens. If the dataset is small or the user is unhappy with it,
read `synthetic-data.md` and consider augmenting from existing goldens.

## Local JSON

```python
from deepeval.dataset import EvaluationDataset

DATASET_PATH = "tests/evals/.dataset.json"

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)
```

## Local JSONL

```python
dataset = EvaluationDataset()
dataset.add_goldens_from_jsonl_file(file_path="tests/evals/.dataset.jsonl")
```

## Local CSV

```python
dataset = EvaluationDataset()
dataset.add_goldens_from_csv_file(file_path="tests/evals/.dataset.csv")
```

If the CSV uses custom column names, set the documented column arguments when
adapting the template.

## Confident AI

```python
dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")
```

Use this when the user says the dataset is on Confident AI and credentials or
MCP/API access are available.

## Pytest Convention

Load the dataset in top-level setup lines, then parametrize with
`dataset.goldens` or `dataset.test_cases`:

```python
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path=DATASET_PATH)


@pytest.mark.parametrize("golden", dataset.goldens)
def test_llm_app(golden):
    ...
```

For end-to-end test cases that are built before assertion, add them back to the
dataset with `dataset.add_test_case(...)`, then parametrize over
`dataset.test_cases` if that better matches the app.

Datasets are either single-turn or multi-turn once loaded. Do not mix `Golden`
and `ConversationalGolden` items in one dataset.

For chatbot / multi-turn agent evals, the loaded dataset contains
`ConversationalGolden`s. After loading, pass `dataset.goldens` to
`ConversationSimulator.simulate(...)` to create `ConversationalTestCase`s for
pytest.
