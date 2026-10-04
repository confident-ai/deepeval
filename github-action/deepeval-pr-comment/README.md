# DeepEval PR Comment (GitHub Action)

A ready-to-use GitHub Action that runs your [DeepEval](https://github.com/confident-ai/deepeval)
tests and posts the evaluation results — totals, per-metric scores, and a
per-test-case breakdown — as a comment on the pull request.

This is the "post eval results as a PR comment" workflow many teams want for
LLM evaluation in CI, without writing custom scripting.

## Usage

Add a workflow file (e.g. `.github/workflows/deepeval.yml`):

```yaml
name: DeepEval

on:
  pull_request:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install deepeval openai

      - name: Run DeepEval and comment
        uses: confident-ai/deepeval/github-action/deepeval-pr-comment@main
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          test_command: "deepeval test run test_eval.py"
```

The action:

1. Runs `test_command` (defaults to `deepeval test run .`).
2. Reads the test-run snapshot DeepEval writes to
   `.deepeval/.latest_run_full.json`.
3. Renders a Markdown summary and posts it as a PR comment using
   `secrets.GITHUB_TOKEN`.

## Inputs

| Input | Default | Description |
| --- | --- | --- |
| `github_token` | _required_ | Token used to post the comment. |
| `test_command` | `deepeval test run .` | Command to run your DeepEval tests. |
| `pr_number` | _auto_ | PR number; auto-detected from `GITHUB_REF` if omitted. |
| `summary_only` | `false` | Only post the metrics summary table. |
| `fail_on_failure` | `false` | Fail the action if any test case failed. |

## Development

The formatter is pure (no network, no DeepEval import) and is unit tested:

```bash
pip install pytest requests
PYTHONPATH=github-action/deepeval-pr-comment python -m pytest github-action/deepeval-pr-comment/tests
```
