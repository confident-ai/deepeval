# Artifact Contracts

Create eval artifacts that users can inspect, edit, commit, and rerun without
an agent.

## Preferred Layout

```text
tests/
  evals/
    test_<app>.py
    .dataset.json
```

Use an existing eval directory if the project already has one.

First look for an existing test folder. If one exists, put the eval suite there.
If none exists, create `tests/evals/`.

Prefer one eval test file for the first setup. Add more files only when the app
needs a separate component-level eval or a clearly distinct use case.

## Dataset Files

Preferred generated dataset path:

```text
tests/evals/.dataset.json
```

Use `.dataset.json`, not `goldens.json`. The mental model is: a dataset contains
goldens.

Supported input formats:

- `.json`
- `.jsonl`
- `.csv`

The dataset should contain the fields needed by the chosen template and metrics.
For RAG, include context or enough information to reconstruct context from the
app. For multi-turn evals, use conversational goldens.

## Pytest Files

Eval tests should:

- load the dataset from `tests/evals/.dataset.json` by default
- call the real app entry point
- build DeepEval test cases
- run a small, explicit end-to-end metric list by default
- add span-level metrics only for useful component diagnostics
- use existing metrics and thresholds when found
- avoid network calls unrelated to the app or evaluation model
- be run with `deepeval test run`, not the raw `pytest` command

## Placeholder Contract

Templates intentionally contain placeholders:

- `TARGET_APP_ENTRYPOINT`
- `DATASET_PATH`
- `EVALUATION_MODEL`
- `METRICS`
- `APP_RESPONSE_ADAPTER`

Replace every placeholder before running evals. If a placeholder remains, stop
and adapt the template instead of running a broken suite.

## Result Artifacts

Do not create hidden result caches unless DeepEval already does so. The durable
artifacts are the test files, dataset files, tracing integration, and optional
Confident AI hosted reports.
