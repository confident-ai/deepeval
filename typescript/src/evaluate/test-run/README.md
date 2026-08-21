# `deepeval test run` — TypeScript port overview

Status snapshot of the TS test-runner integration (`typescript/src/cli/test-run.ts`,
`typescript/src/evaluate/test-run/`, `typescript/src/integrations/vitest/`), how it
differs from Python (`deepeval/cli/test/command.py`, `deepeval/plugins/plugin.py`,
`deepeval/test_run/`), and what's missing. Intended as raw material for the docs and
as the backlog for closing parity.

## Status at a glance

- **The architecture is ported.** `deepeval test run <paths>` calls `startVitest()`
  in-process, the same shape as Python calling `pytest.main()`. Vitest `setupFiles`
  and `globalSetup` stand in for the pytest plugin's per-test and session hooks.
- **`assert_test` is ported as a matcher.** `expect(testCase).toPass([metrics])` and
  `expect(golden).toPass()` cover both of Python's `assert_test` modes — explicit
  test case, and "score the trace this test just produced."
- **Result accumulation is arguably better than Python's.** Each worker appends to
  its own `worker-<pid>.jsonl`, so parallelism needs no file locking, where Python
  serializes every worker through one portalocker-guarded JSON file.
- **Most of the CLI surface is ported.** Eight of Python's dozen-plus flags, with
  the rest either pytest-native or unported. See
  [Cross-cutting gaps](#cross-cutting-gaps).

## Capability matrix

| Capability                      | Python                                      | TypeScript                | Gap                                             |
| ------------------------------- | ------------------------------------------- | ------------------------- | ----------------------------------------------- |
| Runner invocation               | `pytest.main()` in-process                  | `startVitest()` in-process | —                                               |
| Assertion API                   | `assert_test(test_case, metrics)`           | `expect(tc).toPass([...])` | —                                               |
| Trace-scoped assertion          | `assert_test(golden=...)`                   | `expect(golden).toPass()`  | —                                               |
| Parallel workers                | `-n` (xdist) + portalocker                  | Vitest pools + per-pid JSONL | —                                             |
| Plugin auto-registration        | `pytest11` entry point                      | per-file `import "deepeval/vitest"` | [8](#8-no-auto-registration)          |
| Test name in results            | pytest nodeid                               | **`test_case_<n>`**        | [3](#3-the-test-name-is-never-captured)         |
| `--ignore-errors` / `--skip-…`  | Yes                                         | **Hardcoded strict**       | [2](#2-error-handling-is-hardcoded-to-strict)   |
| Metric cache (`-c`)             | `.deepeval-cache.json`                      | **None**                   | [5](#5-no-metric-cache)                         |
| Display filter (`-d`)           | `all` / `failing` / `passing`               | **None**                   | [7](#7-missing-cli-flags)                       |
| Repeat (`-r`), bail (`-x`), `-m`| Yes                                         | **None**                   | [7](#7-missing-cli-flags)                       |
| Batched upload                  | 40 LLM / 20 conversational, POST + PUTs     | **Single POST**            | [4](#4-single-shot-upload-that-fails-silently)  |
| Deferred upload when logged out | `.latest_test_run.json` + `deepeval view`   | **Results deleted**        | [6](#6-results-are-destroyed-when-not-logged-in)|
| Post-run hook                   | `@deepeval.on_test_run_end`                 | **None**                   | [7](#7-missing-cli-flags)                       |
| Hyperparameter logging          | `@deepeval.log_hyperparameters`             | `logHyperparameters()`     | —                                               |
| Run telemetry                   | `capture_evaluation_run(Entrypoint.PYTEST)` | **None**                   | [9](#9-no-run-telemetry)                        |

## Cross-cutting gaps

Ordered by how much they block. The first four are correctness problems; the rest
are unported features.

### 1. `evaluate()` inside a test posts a second, separate test run

Python guards the standalone path so the CLI owns finalization:

```python
# deepeval/evaluate/evaluate.py
if get_is_running_deepeval():
    return EvaluationResult(test_results=test_results, confident_link=None, test_run_id=None)
```

`src/evaluate/evaluate.ts` has no such guard — it calls `postTestRun()`
unconditionally. Anyone who calls `evaluate()` from inside a test file under
`deepeval test run` gets a second TestRun on Confident AI, and those cases never
join the CLI's run. Silent: both runs look plausible.

Needs: a `getIsRunningDeepEval()` early-return in `evaluate()`, with its cases
routed through `globalResultCollector` instead.

### 2. Error handling is configurable (ported)

`run-metrics.ts` reads `--ignore-errors` / `--skip-on-missing-params` out of the
environment via `env-flags.ts`, so a flaky provider call no longer hard-fails a CI
run with no escape hatch.

One cross-SDK divergence stands: Python binds `-i` to `--ignore-errors` and `-id`
to `--identifier`, while the TS CLI keeps `-i` for `--identifier` (changing it
would break existing invocations) and gives `--ignore-errors` no short form.

### 3. The test name is never captured

`buildTestCaseEntry` names each case `testCase.name ?? test_case_${order}`, and
nothing reads the running test's name. Python gets the pytest nodeid via
`PYTEST_RUN_TEST_NAME`. So a suite of well-named tests uploads as `test_case_0`,
`test_case_1`, … and the Confident AI report is unreadable.

Needs: `expect.getState().currentTestName` in the matcher, plumbed into the
persisted entry.

### 4. Single-shot upload that fails silently

`sendTestRun` posts one payload with every case and trace in it. Python batches
(`CONFIDENT_TEST_CASE_BATCH_SIZE`, default 40 LLM / 20 conversational) with a POST
followed by PUTs. A few hundred traced cases will hit a request-size or timeout
limit.

Worse, the failure is swallowed:

```ts
} catch (e) {
  console.warn(`Confident AI: failed to post test run — ${(e as Error).message}`);
```

The process still exits 0, so CI goes green having uploaded nothing.

### 5. Metric cache (ported)

`cache.ts` is the counterpart to `deepeval/test_run/cache.py`: results are keyed by
test-case content plus metric configuration, buffered per process, and merged into
`.deepeval/.deepeval-cache.json` on flush (so sibling Vitest workers don't clobber
each other). `deepeval test run --use-cache` / `-c` turns reads on.

Still missing versus Python: the "Read from Cache" column in the results table.

### 6. Results survive a logged-out run (ported)

`local.ts` writes `.deepeval/.latest_test_run.json` before the upload is attempted,
so a run without `CONFIDENT_API_KEY` is still recoverable and `deepeval view` can
upload it afterwards. `DEEPEVAL_RESULTS_FOLDER` additionally exports a timestamped
`test_run_*.json`.

`deepeval inspect` reads these files directly (see `src/inspect/`), so Python's
separate `.latest_run_full.json` snapshot is not needed — traces already ride
along on each persisted case under `entry.trace`.

### 7. Missing CLI flags

`--official`, `--identifier`, `--ignore-errors`, `--skip-on-missing-params`,
`--verbose`, `--display`, `--max-concurrent` and `--use-cache` are ported. Not
ported, roughly in order of demand:

- `--repeat`, `--exit-on-first-failure`, `--mark` (pytest-native; Vitest has
  native equivalents — `--bail=1`, `-t`, `--reporter`)
- passthrough of extra args to the underlying runner (Python forwards `ctx.args`)
- `@deepeval.on_test_run_end` hook equivalent

`-n` needs nothing — Vitest is parallel by default.

Hyperparameters are ported. `evaluate()` takes `hyperparameters` (and
`identifier`) directly, and `logHyperparameters()` covers the test-run path where
there is no `evaluate()` call to pass them to: each worker writes
`hyperparameters.json` into the shared results dir and `wrapUpTestRun` reads it
back, since a worker cannot reach the main process's memory. The wrap-up warning
in `console-report.ts` is now Python's two-tier version rather than an
unconditional "No hyperparameters logged".

One divergence: Python pushes an unpulled `Prompt` from inside
`process_hyperparameters` to obtain a hash. The TS equivalent is synchronous, so
it warns and drops that entry instead.

### 8. No auto-registration

Python's `pytest11` entry point loads the plugin without the user doing anything.
TS requires `import "deepeval/vitest"` in every test file. The CLI does inject
`setupFiles`, so the import is only needed for the matcher's TypeScript types —
but a missing import is a type error rather than a runtime one, which is a
confusing failure mode.

Related: the CLI passes `setupFiles` and `globalSetup` to `startVitest` as
overrides, which **replace** rather than merge with the project's own config. A
user's setup files (env loading, mocks, fixtures) silently disappear under
`deepeval test run` while working under a plain runner invocation. This should
merge before we document a `vitest.config.ts` setup path in the docs.

### 9. No run telemetry

`src/telemetry.ts` exists but nothing in `src/cli/`, `src/evaluate/test-run/`, or
`src/integrations/vitest/` calls it. Python opens a
`capture_evaluation_run(Entrypoint.PYTEST)` scope in `pytest_sessionstart` with a
run ID shared across workers. We currently have no visibility into TS test-run
adoption at all.

### 10. Trace capture shares one global sink

`beginTraceCapture()` in the Vitest setup calls
`traceManager.setTraceCaptureSink(...)`, the same single global slot that
`evalsIterator` and Mastra's `DeepEvalExporter` write to. Whichever runs last
wins, and the per-test `endTraceCapture()` clears it outright. Calling
`evalsIterator` inside a test file therefore misbehaves. This is
[gap 6 in the integrations README](../../integrations/README.md#6-evalsiterator-and-mastra-fight-over-one-capture-sink),
now with a third consumer — it needs a subscriber list rather than one slot.

### 11. `chatbotRole` is dropped from conversational uploads

The refactor that split `postTestRun` into `buildTestCaseEntry` lost a field the
old conversational branch sent:

```ts
chatbotRole: testCase.chatbotRole,   // present before, absent now
```

One-line fix; listed here so it isn't mistaken for a deliberate omission.

### 12. Result ordering is nondeterministic

`readPersistedCases` concatenates worker files in `readdirSync` order and
`sendTestRun` assigns `order` by array index, so the same suite uploads in a
different order run to run. Python sorts test cases in `wrap_up_test_run` before
display and upload, which is what makes its side-by-side regression view stable.

### 13. Worker env propagation outside the CLI is unverified

`global-setup.mts` sets `DEEPEVAL_RUNNING` and `DEEPEVAL_RESULTS_DIR` on the main
process's `process.env`. Whether workers observe those mutations depends on the
pool and on spawn ordering. If they don't, `getIsRunningDeepEval()` is false in
the worker, nothing is persisted, and wrap-up silently finds zero cases. The CLI
path is safe because `runTest` also passes `env` to `startVitest` — but the
config-file path documented for plain-runner users is not covered by that, and
needs a test.

## Suggested order of work

1. **`evaluate()` guard** (gap 1) — smallest change that stops a silently wrong
   result on the platform.
2. **Test name capture** (gap 3) — one line, and the difference between a readable
   report and `test_case_7`.
3. **Error config flags** (gap 2) — unblocks real CI adoption.
4. **Upload batching + surfacing failures** (gap 4) — stops green CI runs that
   uploaded nothing.
5. **`setupFiles` merge** (gap 8) — prerequisite for documenting a config-file
   setup path.
6. ~~**Local persistence + `view` / `inspect`** (gap 6)~~ — done; both commands
   now read the local files, so the logged-out path is as useful as Python's.
7. ~~**Hyperparameters, then cache**~~ (gaps 7, 5) — both done. The cache key does
   not currently include hyperparameters, which is the remaining loose end.
8. **Capture-sink subscriber list** (gap 10) — shared fix with the integrations
   backlog.
9. **`chatbotRole`, ordering, telemetry, env test** (gaps 11, 12, 9, 13) — small
   independent cleanups.
