# DeepEval Skills

Agent Skills that teach coding assistants how to add DeepEval evaluations,
generate datasets, instrument applications with tracing, and iterate on AI
applications using eval results.

## Skills

| Skill | Description |
| --- | --- |
| [deepeval](./deepeval) | Main DeepEval skill for adding evals to AI apps, generating or reusing datasets, creating pytest eval suites, enabling tracing, sending results to Confident AI, and iterating on failures. |
| [deepeval-otel](./deepeval-otel) | Instrument any app with raw OpenTelemetry so traces export to Confident AI's Observatory — no deepeval package required. Covers the confident.* span/trace attributes and the OTLP endpoint. |
| [deepeval-tracing](./deepeval-tracing) | Instrument an AI app with DeepEval's native tracing — @observe, span types, tags/metadata, and the framework / model / vector-DB integration index — so traces reach Confident AI. |

## Installation

### For Claude.ai (Web)

1. Download the `skills/deepeval` folder from this repository.
2. Zip the folder.
3. In Claude.ai, navigate to **Settings > Capabilities > Skills**.
4. Click **Upload skill** and select your zipped folder.

### For Claude Code (Local CLI)

Download or clone the `skills/deepeval` folder inside the skills folder and place it directly into your local project's skills directory:

```bash
mkdir -p .claude/skills/
cp -r path/to/downloaded/deepeval .claude/skills/
```

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
