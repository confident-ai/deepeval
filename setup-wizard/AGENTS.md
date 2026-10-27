# Agent guide

## Purpose

This directory is DeepEval's setup wizard, released as the `deepeval-setup`
binary from `confident-ai/deepeval` and run by `deepeval setup` (Python) and
`npx deepeval setup` (TypeScript). Keep it focused on creating and running
evaluations, not general tracing setup.

## Commands

```sh
npm run format
npm run lint
npm run typecheck
npm test
npm run build
```

## Constraints

- Keep source TypeScript strict, ESM, modular, and dependency-injected at I/O
  boundaries.
- Never log, serialize, pass as an argument, or send telemetry containing an
  API key or setup token.
- Using Confident AI is always an explicit opt-in, and "No, keep DeepEval
  local" stays a first-class path that makes no Confident AI calls. `--from` is
  only the sign-in and analytics source label.
- An existing `CONFIDENT_API_KEY` (environment first, then `.env.local`, as
  DeepEval resolves it) is offered for reuse before step 1, through
  `src/credentials.ts`. Its check is one authenticated GET that must stay free
  of plan gates: a 401 has to mean the key itself was rejected. The test-run
  routes answer a free plan with 401, so they cannot serve as that check.
- Releases are `setup-v<version>` tags built by
  `.github/workflows/setup_wizard_release.yml` at the repository root, and the
  tag must match the version in `package.json`. Both launchers,
  `deepeval/cli/setup/launcher.py` and `typescript/src/cli/setup/launcher.ts`,
  run the newest published `setup-v*` release, so every release reaches all
  users at once. The archive names `deepeval-setup-<os>-<arch>.tar.gz` and
  `SHA256SUMS` are what both launchers download, so change them together.
- Preserve `.env.local` contents when changing the Confident key or judge-model
  keys, reject symlinks, and retain mode `0600`.
- `src/judge.ts` mirrors DeepEval's own judge resolution: `USE_*` flags first,
  then the OpenAI fallback. Change it only against DeepEval's
  `initialize_model`, `config/settings.py`, and `models/llms/constants.py`, and
  keep judge secrets masked at entry and absent from logs and prompts, which
  name only their variables. A provider's `settings` list only what DeepEval has
  no default for, so never prompt for a value DeepEval would supply itself.
- Network behavior must be mockable; unit tests must not access the network.
- Browser auth and onboarding endpoints are existing API contracts. Validate
  their responses.
- Full-permission agent execution must always follow a separate explicit user
  confirmation.
- Every agent in `src/agents.ts` needs a genuinely read-only invocation for its
  smoke check, a status command, and a `src/agent-progress.ts` parser. Those
  flags are third-party contracts: check them against the vendor's current CLI
  reference rather than from memory, since a wrong flag lands on a real project.
- `prompts/evaluation.md` is the canonical agent prompt. Avoid duplicate prompt
  copies. It mirrors DeepEval's own skills, so keep it aligned with
  `skills/deepeval` and `skills/deepeval-tracing` at the repository root:
  component-level evaluation is the required shape, with metrics attached to the
  spans they judge and any end-to-end metric inside that same suite, and the
  evaluation runs exactly once, since exploratory runs spend judge money and
  each one that reaches Confident AI creates another test run. A suite that
  reports no span level is a failure, enforced by `src/result.ts`. That
  contract runs after the suite has already spent judge money, so a result it
  rejects is reported as unverified with its test run and rerun command
  salvaged, never raised as a wizard failure that discards both.
- Follow the `deepeval` skill's recommendations by default — three to five
  metrics, a `GEval` for whatever the product means by a good answer, metric
  lists in their own module named after the component they score, the
  `tests/evals/` layout with the dataset in `.dataset.json` loaded by the SDK's
  own loader, `deepeval test run` over a bare `pytest`, and its per-shape metric
  defaults. The divergences below are deliberate and each has a reason; anything
  else should match the skill.
- The dataset deliberately diverges from the `deepeval` skill, which reaches for
  `deepeval generate`, targets 30-50 goldens, and forbids hand-written ones.
  Setup curates 20 out of what the repository already knows and calls no model to
  invent them: generation is Python-only, and a first run should not spend judge
  money on data before the user has seen a single score. Reusing a dataset the
  project already has still takes precedence.
- Two more divergences, both because setup is a first run rather than an
  iteration loop: component metrics are required rather than added once
  end-to-end scores point at a component, and the suite runs once instead of the
  skill's five rounds.
- An application in a language neither SDK covers is the one exception, and it
  never gets a fake component tree: it is evaluated black-box, from a standalone
  script that calls the application and scores test cases built from goldens,
  with Python hosting the script rather than being the project's own dependency.
  `shape` in the result keeps the two apart, and each shape's `levels` are
  validated against it.
- Both DeepEval SDKs are in scope, and the prompt carries one spelling table for
  the pair. Read the spellings off `deepeval`'s `typescript/src` and its Python
  package rather than translating one language into the other, since the
  surfaces genuinely differ: synthetic generation, `AsyncConfig`, `CacheConfig`,
  and LiteLLM are Python-only, and `evalsIterator` is async-only.
- A component suite is three metrics on the trace, one of which must be
  `TaskCompletionMetric`, plus one or two on each span worth scoring, so its size
  follows the component map; a black-box suite is five metrics. Twelve is the
  ceiling either way, matching `src/result.ts`. Judge metrics are the default
  whenever a judge is configured, roughly three of them `GEval`s, since judged
  scoring is what the evaluation is for. Configuring a judge is itself the
  agreement to spend on it, so never reintroduce a prompt offering a
  judge-free-only run: it buys nothing and costs the evaluation its point. Judge-free metrics are the
  no-judge fallback, not a cost saving: `ExactMatchMetric`, `PatternMatchMetric`,
  `JsonCorrectnessMetric`, and `ToolCorrectnessMetric` score by comparison or
  validation, though the last two still resolve a model they never ask to score.
  A slot is never a reason to ship a metric that cannot fail, so the prompt lets
  the count fall short and say why instead.
- The evaluation runs once and is never repeated, not even to check a fix: a low
  score is a finding to report. Only a command that failed before any metric
  scored may be issued again, once. The agent then closes with a short report —
  what the run found, what changed, the rerun command, one next step — which must
  not restate what `summarizeResult` in `src/wizard.ts` already prints.
- Multi-turn is built only on evidence from the application's code that
  conversation state crosses turns, never from the product being called a
  chatbot, and it is an addition to the component-level suite. The `thread`
  level follows that same test.
- The prompt's metric table, suite skeletons, and run flags are third-party
  contracts, so check them against the SDKs rather than from memory: metric names
  and the fields each one reads against `_required_params`, the skeletons against
  `skills/deepeval/templates` and `typescript/examples`, and the flags against
  both `test run` commands, whose short forms disagree — Python's `-i` is
  `--ignore-errors` while TypeScript's is `--identifier` — which is why the
  prompt spells every flag out. A metric named there must exist in both SDKs.
- `src/deepeval.ts` decides where DeepEval lives before setup starts, for both
  languages. Its install commands are third-party contracts, so check them
  against Poetry, uv, pip, npm, pnpm, Yarn, and Bun rather than from memory.
  Never install into an interpreter the machine manages: prefer an existing
  environment, otherwise a new project `.venv`. The Node install is a regular
  dependency because component-level instrumentation puts `deepeval/tracing`
  imports in application code.
- `examples/` is git-ignored scratch space for manual end-to-end runs, never
  part of the published package. A sandbox target app there keeps its support
  history unlabeled on purpose: curating evaluation datasets is the work being
  tested, so do not add expected outputs or relevance labels to its data.
- Terminal color must go through `src/theme.ts`, which honors `NO_COLOR` and
  non-TTY output. Brand violet is only the Confident AI wordmark and name;
  everything else the wizard highlights uses the cyan accent, plus ember for
  problems, so do not add hues. Clack's own blue/green/yellow symbols are
  re-themed in `src/ui.ts`; import `log` and `spinner` from there rather than
  from `@clack/prompts`.
- Do not commit or push unless the user explicitly asks.
