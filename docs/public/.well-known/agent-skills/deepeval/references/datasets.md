# Datasets

Use documented `EvaluationDataset` APIs directly. Do not invent wrapper helpers
for dataset loading in templates.

Dataset source order is strict:

1. Ask whether the user already has a dataset.
2. If they do, load it with the documented `EvaluationDataset` API.
3. If they do not, read `synthetic-data.md` and generate one with
   `deepeval generate`.

Do not hand-create or make up goldens. For a useful first generated eval
dataset, target about 30-50 goldens. If the user insists on manual goldens,
warn that generated goldens are usually less biased and more reproducible, then
recommend augmenting any manual seed set with `deepeval generate --method
goldens`.

If the user has a dataset, check its size before accepting it as sufficient.
Fewer than 10 goldens is very likely too small. If the dataset is small or the
user is unhappy with it, read `synthetic-data.md` and consider augmenting from
existing goldens with `deepeval generate`.

## Local JSON

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")
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

Load the dataset directly in the test file immediately before parametrization.
Do not hide dataset loading in `conftest.py` or custom fixture wrappers:

```python
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")

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
