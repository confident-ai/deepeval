# DeepEval Skills

Agent Skills that teach coding assistants how to add DeepEval evaluations,
generate datasets, instrument applications with tracing, and iterate on AI
applications using eval results.

## Skills

| Skill | Description |
| --- | --- |
| [deepeval](./deepeval) | Main DeepEval skill for adding evals to AI apps, generating or reusing datasets, creating pytest eval suites, enabling tracing, sending results to Confident AI, and iterating on failures. |

## Installation

### Cursor Plugin

This repository includes a Cursor plugin manifest that points to `./skills/`.
When installed as a plugin, Cursor can discover the `deepeval` skill directly.

### skills CLI

Install the skill with a skills-compatible installer:

```bash
npx skills add confident-ai/deepeval --skill "deepeval"
```

### Manual Copy

Copy or symlink `skills/deepeval` into your agent's skills directory.

## Prerequisites

For local evals, install DeepEval in the target project:

```bash
pip install -U deepeval
```

For hosted reports, traces, production monitoring, or online evals, connect
DeepEval to Confident AI:

```bash
deepeval login
```
