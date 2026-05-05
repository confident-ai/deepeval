# Synthetic Data

Use `deepeval generate` when the user does not already have a dataset or wants
to augment existing goldens. Generated files should be visible, editable, and
committed with the eval suite when appropriate.

## Choosing a Source

Before generating, ask:

"Do you have documents or knowledge sources I should generate from?"

Prefer this order:

1. Documents or exported retrieval contexts
2. Existing small/weak dataset augmentation
3. Scratch generation

Do not jump straight to scratch if the app has docs, a knowledge base, support
articles, product pages, or exported retrieval contexts.

Use existing-goldens augmentation only when the user says they have a small
dataset, shows dissatisfaction with their current dataset, or you inspect the
dataset and find it is too small or narrow.

## Dataset Size

Check dataset size when a dataset exists. If it has fewer than 10 goldens, treat
it as very likely insufficient and recommend augmentation. A useful first eval
dataset is usually 50-100 goldens. If generation cost or time is a concern,
start smaller but explain that it is a smoke test, not a strong eval set.

## Documents

Use this for RAG apps or apps grounded in docs:

```bash
deepeval generate \
  --method docs \
  --variation single-turn \
  --documents ./docs \
  --output-dir ./tests/evals \
  --file-name .dataset
```

For chatbot or multi-turn agent use cases, generate multi-turn goldens by
default:

```bash
deepeval generate \
  --method docs \
  --variation multi-turn \
  --documents ./docs \
  --scenario-context "Users having multi-turn conversations with the app" \
  --conversational-task "Help users complete their task accurately across turns" \
  --participant-roles "User and assistant" \
  --output-dir ./tests/evals \
  --file-name .dataset
```

Use `--variation single-turn` for chatbot only if the user explicitly asks for
QA pairs for testing for now.

Use multiple document sources by repeating `--documents`:

```bash
deepeval generate \
  --method docs \
  --variation single-turn \
  --documents ./docs \
  --documents ./README.md \
  --documents ./support_articles \
  --output-dir ./tests/evals \
  --file-name .dataset
```

## Contexts

Use this when the project can export retrieval contexts:

```bash
deepeval generate \
  --method contexts \
  --variation single-turn \
  --contexts-file ./tests/evals/contexts.json \
  --output-dir ./tests/evals \
  --file-name .dataset
```

`contexts.json` should be shaped like:

```json
[["chunk 1", "chunk 2"], ["another context chunk"]]
```

## Scratch

Use this when the user has no documents or dataset:

```bash
deepeval generate \
  --method scratch \
  --variation single-turn \
  --num-goldens 20 \
  --scenario "Users asking questions about the app" \
  --task "Answer accurately and concisely" \
  --input-format "Natural language user questions" \
  --output-dir ./tests/evals \
  --file-name .dataset
```

For chatbot or multi-turn agent use cases, default to multi-turn scratch
generation:

```bash
deepeval generate \
  --method scratch \
  --variation multi-turn \
  --num-goldens 20 \
  --scenario-context "Users having multi-turn conversations with the app" \
  --conversational-task "Help users complete their task accurately across turns" \
  --participant-roles "User and assistant" \
  --output-dir ./tests/evals \
  --file-name .dataset
```

For a quick single-turn smoke dataset, keep it small:

```bash
deepeval generate \
  --method scratch \
  --variation single-turn \
  --num-goldens 5 \
  --scenario "Users asking common questions about the app" \
  --task "Answer accurately using the app's normal behavior" \
  --input-format "Short natural language user questions" \
  --output-dir ./tests/evals \
  --file-name .dataset
```

## Existing Goldens

Use this to augment a small user-provided dataset:

```bash
deepeval generate \
  --method goldens \
  --variation single-turn \
  --goldens-file ./tests/evals/.dataset.json \
  --output-dir ./tests/evals \
  --file-name .dataset_augmented
```

Use existing goldens augmentation when the user has a small seed dataset and
wants broader coverage without starting from scratch.

## Model and Cost Options

Pass a generation model when the user chose one:

```bash
deepeval generate \
  --method scratch \
  --variation single-turn \
  --num-goldens 20 \
  --scenario "Users asking common questions about the app" \
  --task "Answer accurately using the app's normal behavior" \
  --input-format "Short natural language user questions" \
  --model gpt-4.1 \
  --cost-tracking \
  --output-dir ./tests/evals \
  --file-name .dataset
```

Use `--cost-tracking` when supported and useful for the user.

## After Generation

Load the generated dataset with documented `EvaluationDataset` APIs:

```python
dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")
```

If the user is not already logged into Confident AI or does not have
`CONFIDENT_API_KEY` exported, ask:

"Do you want to save this generated dataset to Confident AI as well? It is free
of charge and makes it easier to reuse, annotate, and share later."

Options:

- Yes, save it to Confident AI
- Maybe later

If they say yes, authenticate with `deepeval login` for local interactive setup
or `CONFIDENT_API_KEY` for CI/non-interactive setup, then push the dataset:

```python
dataset.push(alias="My Generated Dataset")
```

## Output Contract

Prefer:

```text
tests/evals/.dataset.json
```

Do not store generated goldens only in a hidden cache.
