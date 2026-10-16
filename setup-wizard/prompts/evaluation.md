# Confident AI evaluation setup

You are building a real, rerunnable **component-level DeepEval evaluation** for
the project in the current working directory, and running it once. Component
level means each component of the application — retrieval, each model call, each
tool, each sub-agent — is a span, and the metrics sit on the spans they judge, so
a failing run names the component that failed instead of only the answer. This is
evaluation work, not a tracing integration.

Work through the steps in order. Every choice comes out of this repository, so
read before you write, and do not invent an SDK call, a metric, or a domain.

## Safety and boundaries

- Never open, print, parse, search, or otherwise read `.env.local`.
- When the runtime context says Confident AI is configured, `CONFIDENT_API_KEY`
  is already available to DeepEval through the process environment or the
  project dotenv. Use it only through the SDK; never copy it into source,
  output, commands, logs, or result files.
- When the runtime context says the run is local-only, do not upload to
  Confident AI, do not read or set `CONFIDENT_API_KEY`, and do not produce a
  test run ID.
- Judge-model credentials reach DeepEval the same way. The runtime context below
  names the configured provider; select metrics that its judge can serve, and
  never read, print, or hardcode provider keys. When no judge is configured,
  ship judge-free metrics only.
- The user already agreed to this run by configuring a judge and starting setup,
  so do not ask again about provider usage and do not offer to hold the judge
  metrics back. A configured judge means every metric runs.
- Instrument the application only as far as the evaluation needs it: the
  components being scored, and nothing else. Production monitoring, dashboards,
  and trace tagging for observability are out of scope.
- Never run `deepeval login`; credentials are already configured.
- Make focused changes only. Do not commit or push.

## Step 1 — Read the repository and write down the component map

Do this before editing anything.

1. **Settle the language.** When the runtime context names an SDK, use it.
   Otherwise: `pyproject.toml`, `requirements.txt`, `setup.py`, or `Pipfile`
   mean the Python SDK, and `package.json` means the TypeScript SDK. If both are
   present, follow the code that actually calls the model. If neither is, the
   application cannot be instrumented — jump to
   [When the project is in neither language](#when-the-project-is-in-neither-language).
2. **Find the entry point:** the function, route, or command that takes one user
   request and returns the answer. The evaluation calls exactly this.
3. **Write the component map.** List every component the entry point reaches
   that can fail on its own, and for each one record four things: the function,
   the span type, what it receives, and what it returns.

   | Span type   | What it is                                        |
   | ----------- | ------------------------------------------------- |
   | `agent`     | the entry point, and each sub-agent or delegation |
   | `retriever` | a search over a corpus, vector store, or index    |
   | `llm`       | one model call                                    |
   | `tool`      | one function the model can choose to call         |

   Anything that is not one of those four — a formatter, a database write, an
   HTTP handler — is not a component worth scoring. Do not observe it.

4. **Record what reference data this repository can supply.** For a golden, can
   it state the right answer, the documents that should have been retrieved, the
   tools that should have fired? Is the output structured enough to validate
   against a schema, or does any part of a correct answer have to match a fixed
   string or pattern? This decides which metrics are available in step 4, and
   especially how many judge-free ones, so answer it now rather than discovering
   it at runtime.
5. **Classify the application**, in one word, because it decides the starting
   metrics in step 4: multi-turn, agent, RAG, or plain LLM, in that precedence —
   an application that both retrieves and uses tools is an agent, and one that
   does either across conversation turns is multi-turn. Multi-turn is the one
   label the code has to earn, by the test in
   [Single-turn or multi-turn](#single-turn-or-multi-turn); assume single-turn
   until the entry point shows conversation state crossing turns.
6. **Reuse what exists.** If the repository already has DeepEval metrics,
   thresholds, datasets, or an evaluation directory, extend them instead of
   building a parallel setup, and keep its thresholds unless they are unset.

## Step 2 — Install DeepEval

Confirm DeepEval is installed for the language from step 1, and install it with
the project's own package manager if it is missing: the Python SDK into the
project's own environment on Python 3.9 or newer, the TypeScript SDK into the
project's `package.json`. If it cannot be installed, stop and report `failed`
with the reason.

## Step 3 — Build a dataset of 20 goldens

Hold the goldens in DeepEval's own `EvaluationDataset`, never in a plain list or
a bespoke JSON shape the suite parses itself.

- If the project already has a dataset, use it — including one on Confident AI,
  which both SDKs pull by alias.
- Otherwise write **exactly 20 goldens** yourself, sourced from what the
  repository already knows: its docs, fixtures, prompts, sample requests, seed
  data, and existing tests. Every input must be one a real user of this
  application would send, and the 20 together should cover its ordinary traffic
  plus the edges it is most likely to get wrong. Do not call a model to invent
  them, and do not invent a domain the repository does not have.
- Every golden needs an `input`. Add `expected_output`, `context`,
  `retrieval_context`, or `expected_tools` **only where step 1 said the
  repository actually knows them.** A metric that reads a missing field fails at
  runtime, and a guessed reference is worse than an absent one: it scores the
  application against fiction.
- Write them to `.dataset.json` beside the suite — that name, not
  `goldens.json`, because a dataset is the thing that holds goldens — and load
  it with the SDK's own loader rather than parsing it yourself. It is a JSON
  array of objects whose keys are the snake_case golden field names, `input`,
  `expected_output`, `context`, `retrieval_context`, `expected_tools`, which is
  what both SDKs expect by default, so one file serves either language.

## Step 4 — Put three metrics on the trace and one or two on each span

Every metric named here exists in both SDKs. "Reads" is what the metric takes
off the span it is attached to, and a metric whose fields the span does not carry
fails the run, so check each one against your component map and your goldens.
Judge metrics call the configured model on every golden; judge-free metrics score
by comparison or validation, spending nothing and needing no provider key.

| Metric                                        | Judge-free | Reads                                     | Attach to                                    |
| --------------------------------------------- | ---------- | ----------------------------------------- | -------------------------------------------- |
| `ExactMatchMetric`                            | yes        | input, output, expected output            | the span whose answer is exact               |
| `PatternMatchMetric`                          | yes        | input, output, plus a `pattern` you write | any span with text output                    |
| `JsonCorrectnessMetric`                       | yes        | input, output, plus an `expected_schema`  | the span that returns structured output      |
| `ToolCorrectnessMetric`                       | yes        | input, tools called, expected tools       | the `agent` span that calls tools            |
| `ContextualRelevancyMetric`                   | no         | input, retrieval context                  | the `retriever` span                         |
| `ContextualRecallMetric`                      | no         | input, retrieval context, expected output | the `retriever` span                         |
| `ContextualPrecisionMetric`                   | no         | input, retrieval context, expected output | the `retriever` span                         |
| `AnswerRelevancyMetric`                       | no         | input, output                             | the `llm` span that answers                  |
| `FaithfulnessMetric`                          | no         | input, output, retrieval context          | the `llm` span that answers                  |
| `ArgumentCorrectnessMetric`                   | no         | input, tools called                       | the `agent` span that calls tools            |
| `TaskCompletionMetric`                        | no         | input, output                             | the trace, required on every component suite |
| `PlanAdherenceMetric`, `StepEfficiencyMetric` | no         | input, output                             | the trace, for trajectory                    |
| `GEval`                                       | no         | whichever fields you list in it           | any span whose criterion is product-specific |

**The budget.** A component-level suite carries **three metrics on the trace and
one or two on every span worth evaluating**, so its size follows the component
map rather than a fixed number. One of the three trace metrics must be
`TaskCompletionMetric`: whether the application did what was asked is the
question every other score exists to explain. A black-box suite has no spans, so
it is **five metrics** in total. Twelve is the hard ceiling either way — on an
application with more components than that allows, score the ones on the path
from the request to the answer and leave the rest unscored.

**Judge metrics are the point when a judge is available.** With a judge
configured, every metric should be a judge metric, and about three of them
`GEval`s written for this product. Judged scoring is what makes an evaluation say
something a unit test could not, so do not trade it away for cheapness the user
did not ask for.

**Judge-free metrics are the fallback, not the default.** With no judge
configured the suite is judge-free in full and `errors` says so. A judge-free
metric still has to be able to fail: `PatternMatchMetric`
needs a `pattern` a wrong answer would miss, `JsonCorrectnessMetric` a schema the
output really claims to satisfy, `ExactMatchMetric` an answer that is genuinely
exact, `ToolCorrectnessMetric` the tools a golden knows should fire. Never write a
pattern that matches everything to fill a slot; report the shortfall instead.

The rest of the rules:

- Write around three `GEval`s, because a product's idea of a good answer is
  rarely a built-in metric. There is no generic correctness metric — correctness
  depends on the task — so name each one for something this application owes its
  user and put the criterion in its own words.
  `GEval(name="Correctness", criteria="...", evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT], threshold=0.7)`
  in Python, `new GEval({ name, criteria, evaluationParams: [...], threshold: 0.7 })`
  in TypeScript, with `SingleTurnParams` imported from the test-case module. List
  only fields the span will actually carry, preferring input and output over the
  reference fields unless your goldens really supply them. Reach for `DAGMetric`
  only when the score has to follow explicit branches rather than a judgment.
- The trace's other two metrics judge the run as a whole: `GEval` for what the
  product promises, or `StepEfficiencyMetric` and `PlanAdherenceMetric` when the
  trajectory rather than the answer is what you doubt.
- Each span's one or two metrics are that component's own failure modes, and
  retrieval quality belongs on the retriever, never on the generator. A component
  whose output nothing can meaningfully judge gets no metric rather than a
  padded one.
- One list per component, named after it: `RETRIEVER_SPAN_METRICS`,
  `GENERATOR_SPAN_METRICS`, `ORDER_LOOKUP_TOOL_SPAN_METRICS`. Never one shared
  list, and never the same metric on two components.
- Set `threshold` explicitly on every metric. Judge metrics default to `0.5`; use
  `0.7` unless the repository already sets one. Judge-free metrics are pass or
  fail and keep their own default of `1`.

The application's shape from step 1 says which span metrics to start from:

| Application    | Span metrics to start from                                                                                       |
| -------------- | ---------------------------------------------------------------------------------------------------------------- |
| RAG            | `ContextualRelevancyMetric` on the retriever, `FaithfulnessMetric` and `AnswerRelevancyMetric` on the generator  |
| Agent or tools | `ArgumentCorrectnessMetric` on the agent span, `ToolCorrectnessMetric` where the goldens know the expected tools |
| Plain LLM      | `AnswerRelevancyMetric` on the one `llm` span, plus a `GEval` for what the product promises                      |
| Multi-turn     | conversational metrics only, per [Single-turn or multi-turn](#single-turn-or-multi-turn)                         |

## Step 5 — Instrument, and report each span's own fields

Prefer a native DeepEval integration when the project uses a framework DeepEval
supports, and observe the application's own functions where none does. Inside
each observed component, report the fields that component actually has:

| Component | Report on the span                                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------------- |
| retriever | the query as input, the retrieved chunks as retrieval context                                                        |
| llm       | the prompt or question as input, the completion as output                                                            |
| tool      | the arguments as input, the return value as output                                                                   |
| agent     | the request as input, the final answer as output, plus tools called and expected tools when a tool metric needs them |

Report those fields directly on the span. Only build an `LLMTestCase` for a span
that has an actual output to put in it: a retriever span has none, so
constructing one there fails.

## Step 6 — Write the suite

Two source files, next to the dataset from step 3, in the project's existing
evaluation directory or in `tests/evals/`. Metric lists live in the metrics
module, never inline in the suite, and the metrics import is a plain module
import rather than a relative one, since an eval directory is not a package.

Python — `metrics.py`, then `test_<app>.py`:

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

RETRIEVER_SPAN_METRICS = [ContextualRelevancyMetric(threshold=0.7)]
GENERATOR_SPAN_METRICS = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
]
```

```python
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_span, update_current_trace

from metrics import GENERATOR_SPAN_METRICS, RETRIEVER_SPAN_METRICS

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/.dataset.json")


@observe(type="retriever", metrics=RETRIEVER_SPAN_METRICS)
def retrieve(query: str) -> list[str]:
    chunks = search_docs(query)  # the application's own retrieval
    update_current_span(input=query, retrieval_context=chunks)
    return chunks


@observe(type="llm", metrics=GENERATOR_SPAN_METRICS)
def generate(query: str, chunks: list[str]) -> str:
    answer = call_model(query, chunks)  # the application's own model call
    update_current_span(input=query, output=answer, retrieval_context=chunks)
    return answer


@observe(type="agent")
def run_app(query: str) -> str:
    answer = generate(query, retrieve(query))
    update_current_trace(input=query, output=answer)
    return answer


@pytest.mark.parametrize("golden", dataset.goldens)
def test_app(golden: Golden):
    run_app(golden.input)
    assert_test(golden=golden)  # trace metrics, if any, go here as `metrics=[...]`
```

TypeScript — `metrics.ts`, then `<app>.test.ts`:

```typescript
import {
  AnswerRelevancyMetric,
  ContextualRelevancyMetric,
  FaithfulnessMetric,
} from "deepeval/metrics";

export const RETRIEVER_SPAN_METRICS = [
  new ContextualRelevancyMetric({ threshold: 0.7 }),
];
export const GENERATOR_SPAN_METRICS = [
  new AnswerRelevancyMetric({ threshold: 0.7 }),
  new FaithfulnessMetric({ threshold: 0.7 }),
];
```

```typescript
import { it, expect } from "vitest";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import {
  observe,
  updateCurrentSpan,
  updateCurrentTrace,
} from "deepeval/tracing";
import { GENERATOR_SPAN_METRICS, RETRIEVER_SPAN_METRICS } from "./metrics";
import "deepeval/vitest";

const dataset = new EvaluationDataset();
// Awaited at module scope so the goldens exist when Vitest collects the cases.
await dataset.addGoldensFromJSON({ filePath: "tests/evals/.dataset.json" });

const retrieve = observe({
  type: "retriever",
  metrics: RETRIEVER_SPAN_METRICS,
  fn: async (query: string): Promise<string[]> => {
    const chunks = await searchDocs(query); // the application's own retrieval
    updateCurrentSpan({ input: query, retrievalContext: chunks });
    return chunks;
  },
});

const generate = observe({
  type: "llm",
  metrics: GENERATOR_SPAN_METRICS,
  fn: async (query: string, chunks: string[]): Promise<string> => {
    const answer = await callModel(query, chunks); // the application's own model call
    updateCurrentSpan({
      input: query,
      output: answer,
      retrievalContext: chunks,
    });
    return answer;
  },
});

const runApp = observe({
  type: "agent",
  fn: async (query: string): Promise<string> => {
    const answer = await generate(query, await retrieve(query));
    updateCurrentTrace({ input: query, output: answer });
    return answer;
  },
});

// The empty array is the trace-level metrics; the span metrics come from the
// spans themselves. `task` must start the run, so the trace is not missed.
it.each(dataset.goldens as Golden[])("answers: $input", async (golden) => {
  await expect(golden).toPass([], { task: (g) => runApp(g.input) });
});
```

`search_docs`, `call_model`, and the dataset path stand in for this project's own
retrieval, model call, and file: observe the real functions from the component
map instead, or let a framework integration emit those spans and stage the
metrics onto them. Every placeholder has to be gone before the command runs,
since the one run is the only one you get. What has to survive the substitution
is the shape — a span per component, that component's metrics on it, that
component's fields reported inside it, and the entry point called once per
golden. Do not rebuild a traced suite as hand-made `LLMTestCase`s; that is the
black-box shape, and it throws away the component scores this asks for. In a
script rather than a test, the loop is
`for golden in dataset.evals_iterator()` in Python and
`for await (const golden of dataset.evalsIterator())` in TypeScript, calling the
same observed entry point inside.

## Step 7 — Run it exactly once

Document one command that reruns the whole evaluation, and run that command:
`deepeval test run <path>` in Python, `npx deepeval test run <path>` in
TypeScript, never a bare `pytest` or `vitest` — the wrapper is what opens the
test run and posts the results. Pass `--identifier` to name the run, and write
every flag in long form, because the two CLIs disagree on the short ones: in
Python add `--num-processes 4` to evaluate the 20 goldens concurrently, and in
TypeScript add `--max-concurrent 4`. Add `--skip-on-missing-params` and
`--ignore-errors` too, so one unusable golden or one transient judge error costs
you that case instead of the whole run — but they hide exactly what you must
report, so read the summary and name every skip and error in `errors`, and report
`failed` if nothing scored at all. If Confident AI is configured, this command
also uploads the test run, and the `testRunId` you report must come from this one
run.

Run every metric in the suite, judge metrics included. Finishing without running
it is not an acceptable outcome.

One run, and one only. Once the command has scored the goldens, that run is the
result: a low score is a finding to report, never a reason to run it again with
different thresholds, fewer goldens, or a repaired prompt. Do not verify a fix by
running it, do not run a slice of the suite, and do not run an ad-hoc copy of it
to see how the SDK behaves — judge metrics spend money on every call, and every
run that reaches Confident AI creates another test run for the user to sift
through. When an API is unclear, read the installed package's type definitions or
source instead of calling it. The single exception is a command that fails before
any metric scores, which produced no test run at all: fix its cause and issue the
command once more, at most once, then report what happened either way.

## Step 8 — Tell the user what happened

Finish with a short report, in four parts and under ten lines total. The wizard
already prints the status, dataset, metric names, and rerun command from your
result file, so do not restate them or dump a score table.

1. **What the evaluation found.** The headline from the one run: how many goldens
   passed, and which component came out weakest, with its metric and score. If
   everything passed, say so and name the thinnest coverage instead.
2. **What you changed.** One line per file you added or edited, saying what it
   does — the suite, the metrics, the dataset, and any instrumentation added to
   the application itself.
3. **How to run it again.** The rerun command, on its own line.
4. **The next step.** One sentence: the single most useful thing to do with this
   result, drawn from what the run actually showed — raise the weakest
   component's quality, widen the 20 goldens, or replace a judge-free metric that
   never fails.

Plain sentences, no headings, no secrets, and nothing about how you built it.

## Reference: the two SDKs

Same design, different spelling. Use one column, never both. Do not port a
spelling across: the surfaces genuinely differ, and synthetic generation and the
async and cache config objects are Python-only.

| Purpose                            | Python                                                                                       | TypeScript                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Modules                            | `deepeval.tracing`, `deepeval.metrics`, `deepeval.test_case`, `deepeval.dataset`             | `deepeval/tracing`, `deepeval/metrics`, `deepeval/test-case`, `deepeval/dataset`             |
| Dataset object                     | `EvaluationDataset(goldens=[Golden(input=...)])`                                             | `new EvaluationDataset({ goldens: [new Golden({ input })] })`                                |
| Optional golden fields, in code    | `expected_output`, `context`, `retrieval_context`, `expected_tools`                          | `expectedOutput`, `context`, `retrievalContext`, `expectedTools`                             |
| The same fields in `.dataset.json` | snake_case in both SDKs: `expected_output`, `context`, `retrieval_context`, `expected_tools` | identical to Python, which is why one file serves either language                            |
| Observe a component                | `@observe(type="llm", metrics=[...])`                                                        | `observe({ type: "llm", metrics: [...], fn })`                                               |
| Report a span's fields             | `update_current_span(input=..., output=..., retrieval_context=...)`                          | `updateCurrentSpan({ input, output, retrievalContext })`                                     |
| Report the trace's fields          | `update_current_trace(input=..., output=...)`                                                | `updateCurrentTrace({ input, output })`                                                      |
| Metrics onto an integration's span | `next_llm_span(...)`, and the agent, tool, and retriever forms, as context managers          | `nextLlmSpan(...)`, and the same forms, taking the call as a callback                        |
| Loop over goldens                  | `for golden in dataset.evals_iterator(...)`                                                  | `for await (const golden of dataset.evalsIterator({ ... }))`                                 |
| Assert in a test                   | `assert_test(golden=golden)`                                                                 | `await expect(golden).toPass([], { task })`, with `import "deepeval/vitest"` for the matcher |
| Suite layout                       | `tests/evals/test_<app>.py`, `metrics.py`, `.dataset.json`                                   | `tests/evals/<app>.test.ts`, `metrics.ts`, `.dataset.json`                                   |
| Load the dataset file              | `dataset.add_goldens_from_json_file(file_path=...)`                                          | `await dataset.addGoldensFromJSON({ filePath })`                                             |
| Run the suite                      | `deepeval test run <path>`                                                                   | `npx deepeval test run <path>`                                                               |

## Single-turn or multi-turn

Multi-turn is a separate shape, and whether to build it is decided by the code
rather than by what the product is called. Read the entry point: an application
is multi-turn only when it carries conversation state across turns — a thread,
conversation, or session id it keys on, prior messages passed back into the
model, a store it reads history from. A single-request endpoint or one-shot
function is single-turn even when the product is a chatbot, and a client that
happens to keep a transcript is not evidence: the state has to reach the code
being evaluated. When the code shows none of it, build the single-turn suite and
nothing else, and leave `thread` out of the reported levels. When it does, add
conversational goldens and conversational metrics on top of the component-level
suite, never as a replacement for it.

The two metric families do not mix. A conversational test case takes
conversational metrics — `ConversationCompletenessMetric`, `RoleAdherenceMetric`,
`TurnRelevancyMetric`, `ConversationalGEval` — and never `AnswerRelevancyMetric`,
`FaithfulnessMetric`, or any other single-turn metric. Single-turn spans keep the
single-turn metrics from step 4.

## When the project is in neither language

An application in Go, Java, Ruby, C#, or anything else DeepEval has no SDK for
cannot be instrumented: it emits no spans, so component metrics have nothing to
attach to. Do not port the application, do not add a wrapper in another language
to pose as its components, and do not observe a function that merely forwards a
request — that produces a fake component tree that scores nothing real. Evaluate
it from the outside, with the Python SDK as a harness rather than as one of the
project's dependencies:

1. Build the 20 goldens exactly as in step 3.
2. Call the application the way its own users do, from a standalone script: its
   HTTP endpoint, its CLI, or its entry point as a subprocess. Read the endpoint
   or command out of the repository rather than assuming one, and start whatever
   the call needs. If it cannot be reached without credentials or services you
   do not have, stop and report `failed` naming what was missing.
3. Build one `LLMTestCase` per golden from the answer that comes back, filling
   only the fields the application actually returns.
4. Score them in one pass with `evaluate(test_cases=..., metrics=...)` plus an
   identifier, using the black-box budget from step 4: five metrics, judge
   metrics with about three `GEval`s when a judge is available, and limited to
   those whose fields these test cases actually carry. `TaskCompletionMetric` is
   required of component suites, not this one, since there is no trace to put it
   on.
5. Keep the script and its metrics module together, in the project's existing
   evaluation directory or in `evals/` as `evaluate_<app>.py`, `metrics.py`, and
   `.dataset.json`. The documented rerun command is that script run by the Python
   that holds DeepEval, such as `.venv/bin/python evals/evaluate_<app>.py`, and
   it runs exactly once under the rules in step 7.
6. Report `shape` as `black-box` and `levels` as `test-case`, plus `thread` only
   if the code met the multi-turn test. Do not report `span`: there are none.

## Required result

The environment variable `CONFIDENT_SETUP_RESULT_FILE` points to the only result
file you should write. Before finishing, write valid UTF-8 JSON there with this
exact shape:

```json
{
  "status": "completed",
  "shape": "component-level",
  "changedFiles": ["relative/path"],
  "sdks": ["deepeval-python"],
  "levels": ["test-case", "span", "trace", "thread"],
  "datasetSource": "where the goldens came from and how many",
  "metrics": ["metric names"],
  "rerunCommand": "exact safe command",
  "testRunId": "id returned by the completed Confident AI run",
  "testRunUrl": "optional URL returned by tooling",
  "errors": ["only for skipped metrics, partial work, or failures"]
}
```

Allowed `status` values are `completed`, `partial`, and `failed`. Allowed SDKs
are `deepeval-python` and `deepeval-typescript`, and exactly one of them belongs
in `sdks`. Allowed levels are `test-case`, `span`, `trace`, and `thread`.
`changedFiles`, `sdks`, `levels`, `datasetSource`, `metrics`, and `rerunCommand`
are always required.

`shape` is `component-level` or `black-box`, and it decides what `levels` may
say. A `component-level` result other than `failed` must list both `span` and
`trace`, because that shape scores components and scores the run as a whole; a
suite missing either is not the evaluation this asked for, so report `failed` with
the reason rather than a run without them. A `black-box` result must not list
`span`, and belongs only to an application in a language neither SDK covers.
`shape` defaults to `component-level` when omitted.

A completed result requires `testRunId` when Confident AI is configured; omit
`testRunId` and `testRunUrl` for a local-only run. A failed result requires at
least one `errors` entry. Use repository-relative paths and never include secret
values.

Report `completed` once the evaluation has actually run, naming any metric you
had to skip in `errors`. Reserve `partial` for an implementation that is
rerunnable but could not be executed at all, and never invent a test run ID.
