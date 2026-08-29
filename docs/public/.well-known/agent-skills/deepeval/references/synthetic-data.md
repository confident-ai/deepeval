# Synthetic Data

Use `deepeval generate` when the user does not already have a dataset or wants
to augment existing goldens. Do not hand-create or make up goldens. Generated
files should be visible, editable, and committed with the eval suite when
appropriate.

## Choosing a Source

Before generating, ask:

"Do you already have a dataset?"

If the answer is no, inspect or ask what source material is available and choose
the best `deepeval generate` method. Prefer this order:

1. Existing user-provided dataset
2. Documents, knowledge bases, support articles, product pages, docs folders, or
   READMEs with `deepeval generate --method docs`
3. Exported retrieval contexts with `deepeval generate --method contexts`
4. Existing small/weak dataset augmentation with `deepeval generate --method goldens`
5. Scratch generation with `deepeval generate --method scratch`

Documents and knowledge bases are the best generation source. Do not jump
straight to scratch if the AI app has docs, a knowledge base, support articles,
product pages, READMEs, or exported retrieval contexts.

If the user insists on manually writing goldens, push back once:

"I recommend using `deepeval generate` instead of hand-writing goldens so the
dataset is larger, less biased, and easier to reproduce. If you still want to
manually author a small seed dataset, I can help structure it, but we should
augment it with `deepeval generate --method goldens` before relying on it."

Use existing-goldens augmentation only when the user says they have a small
dataset, shows dissatisfaction with their current dataset, or you inspect the
dataset and find it is too small or narrow.

## Styling Defaults

Always infer the AI app's use case before generating goldens and pass styling
flags by default. This applies to all generation methods: docs, contexts,
goldens, and scratch. Scratch requires the core styling flags, but the other
methods should still use them because styling makes generated goldens more
accurate and specific to the user's AI app.

For single-turn generation, infer and pass:

- `--scenario`: who the users are and what situation they are in
- `--task`: what the AI app should accomplish
- `--input-format`: what realistic inputs look like
- `--expected-output-format`: what a good expected output should look like, if
  expected outputs are generated

For multi-turn generation, infer and pass:

- `--scenario-context`: the conversation setting and user situation
- `--conversational-task`: what the AI app should accomplish across turns
- `--participant-roles`: who participates in the conversation
- `--scenario-format`: what generated scenarios should look like
- `--expected-outcome-format`: what a successful conversation outcome should
  look like, if expected outcomes are generated

If the use case is not clear from the codebase or docs, ask one concise
question:

"What does your AI app do, who uses it, and what kinds of inputs should the eval
dataset cover?"

## Dataset Size

Check dataset size when a dataset exists. If it has fewer than 10 goldens, treat
it as very likely insufficient and recommend augmentation. A useful first
generated eval dataset should usually have about 30-50 goldens. If generation
cost or time is a concern, start smaller but explain that it is a smoke test,
not a strong eval set.

## Documents

Use this for RAG apps or apps grounded in docs:

```bash
deepeval generate \
  --method docs \
  --variation single-turn \
  --documents ./docs \
  --num-goldens 40 \
  --scenario "Users relying on the AI app for product-specific help" \
  --task "Help users complete their task accurately using the available documentation" \
  --input-format "Natural language requests with product-specific details" \
  --expected-output-format "Concise, actionable output grounded in the provided documents" \
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
  --num-goldens 40 \
  --scenario-context "Users having multi-turn conversations with the app" \
  --conversational-task "Help users complete their task accurately across turns" \
  --participant-roles "User and assistant" \
  --scenario-format "A realistic conversation scenario with product-specific constraints" \
  --expected-outcome-format "The user reaches a correct, actionable resolution grounded in the documents" \
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
  --num-goldens 40 \
  --scenario "Users relying on the AI app for product-specific help" \
  --task "Help users complete their task accurately using the available documentation" \
  --input-format "Natural language requests with product-specific details" \
  --expected-output-format "Concise, actionable output grounded in the provided documents" \
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
  --num-goldens 40 \
  --scenario "Users relying on the AI app for context-grounded help" \
  --task "Help users complete their task accurately using retrieved context" \
  --input-format "Natural language requests that should be answered from retrieved context" \
  --expected-output-format "Concise, actionable output grounded in the provided contexts" \
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
  --num-goldens 40 \
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
  --num-goldens 40 \
  --scenario-context "Users having multi-turn conversations with the app" \
  --conversational-task "Help users complete their task accurately across turns" \
  --participant-roles "User and assistant" \
  --output-dir ./tests/evals \
  --file-name .dataset
```

Only create a tiny smoke dataset when the user explicitly asks for a smoke test.
Otherwise generate about 30-50 goldens:

```bash
deepeval generate \
  --method scratch \
  --variation single-turn \
  --num-goldens 10 \
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
  --num-goldens 40 \
  --scenario "Users represented by the existing seed dataset" \
  --task "Expand coverage while preserving the AI app's intended behavior" \
  --input-format "Inputs similar in style and structure to the seed goldens" \
  --output-dir ./tests/evals \
  --file-name .dataset_augmented
```

Use existing goldens augmentation when the user has a small seed dataset and
wants broader coverage without starting from scratch. Do not write the extra
goldens by hand.

## Model and Cost Options

Pass a generation model when the user chose one:

```bash
deepeval generate \
  --method scratch \
  --variation single-turn \
  --num-goldens 40 \
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
