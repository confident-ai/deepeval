// Batched System One for `evaluate()`: every metric Jev judges on a test case,
// asked in as few requests as the token budget allows. Mirrors Python's
// `deepeval.metrics.utils.system_one_batch`.
//
// Under `system_one` a metric's own `measure` sends one request per metric
// (`runSystemOneEval`), re-sending the same test case each time. Jev evaluates
// every question in a request independently against one shared state, so the
// executors instead collect a test case's `system_one` metrics, merge their
// states and questions into one request, and hand each metric back its own
// answers and share of the cost. Scores, reasons and confidence are filled in
// by the same `settle` a lone measure uses.
//
// A metric is batched when its effective eval mode is `system_one` (so an
// `evalMode` option on the metric is honoured), it has a whole-metric form for
// the test case, and the caller has no cached result for it. With fewer than
// two such metrics there is nothing to share, and every metric runs its own
// `measure`.

import { EvalMode } from "@/config/eval-mode";
import { MissingTestCaseParamsError } from "@/errors";
import { BaseMetric, startMeasure } from "@/metrics/base-metrics";
import { BaseConversationalMetric } from "@/metrics/base-conversational-metric";
import { prepareMeasure } from "@/metrics/prepare-measure";
import {
  contextLimitError,
  effectiveEvalMode,
} from "@/metrics/system-one/decision";
import {
  requestFor,
  settle,
  type SystemOneEvalSpec,
} from "@/metrics/system-one/runner";
import {
  SystemOneContextLimitError,
  checkContextBudget,
} from "@/models/system-one/limits";
import {
  SystemOneAnswers,
  type SystemOneQuestion,
} from "@/models/system-one/schema";
import { ConversationalTestCase, LLMTestCase } from "@/test-case";

type AnyMetric = BaseMetric | BaseConversationalMetric;
type State = Record<string, unknown>;

export interface SystemOneBatchOptions {
  ignoreErrors: boolean;
  skipOnMissingParams: boolean;
  /**
   * Show the batch as one line of the caller's progress display; returns the
   * callback that clears it. Without it, a batch of metrics that would each
   * show a spinner shows a single spinner instead.
   */
  onStart?: (label: string) => () => void;
}

///////////////////////////////////////////////
// Requests
///////////////////////////////////////////////

interface Entry {
  metric: AnyMetric;
  spec: SystemOneEvalSpec;
  state: State;
  questions: Record<string, SystemOneQuestion>;
}

/**
 * One Jev call. Questions are keyed `m{j}.{key}` so each entry's answers can be
 * picked back out; keys are never sent to the model.
 */
class JevRequest {
  constructor(
    public entries: Entry[],
    public state: State,
  ) {}

  get model() {
    return this.entries[0].metric.systemOneModel!;
  }

  get questions(): Record<string, SystemOneQuestion> {
    const questions: Record<string, SystemOneQuestion> = {};
    this.entries.forEach((entry, j) => {
      for (const [key, question] of Object.entries(entry.questions)) {
        questions[`m${j}.${key}`] = question;
      }
    });
    return questions;
  }

  /**
   * Add `entry` when it asks the same Jev model, its state merges without
   * conflict and the merged request still fits Jev's budget.
   */
  admit(entry: Entry): boolean {
    const modelName = entry.metric.systemOneModel!.getModelName();
    if (modelName !== this.model.getModelName()) return false;
    const state = mergeStates(this.state, entry.state);
    if (state === undefined) return false;
    const merged = new JevRequest([...this.entries, entry], state);
    try {
      checkContextBudget(merged.state, merged.questions);
    } catch (err) {
      if (err instanceof SystemOneContextLimitError) return false;
      throw err;
    }
    this.entries = merged.entries;
    this.state = merged.state;
    return true;
  }

  answersFor(answers: SystemOneAnswers, j: number): SystemOneAnswers {
    const prefix = `m${j}.`;
    const pick = <T>(byKey: Record<string, T>): Record<string, T> =>
      Object.fromEntries(
        Object.entries(byKey)
          .filter(([key]) => key.startsWith(prefix))
          .map(([key, answer]) => [key.slice(prefix.length), answer]),
      );
    return new SystemOneAnswers({
      nouls: pick(answers.nouls),
      choices: pick(answers.choices),
      scores: pick(answers.scores),
    });
  }
}

/**
 * `a` and `b` as one state, or `undefined` when they give the same top-level
 * key different values. `test_case` fields and each turn's fields always
 * merge: both states were built from the same test case.
 */
function mergeStates(a: State, b: State): State | undefined {
  const merged: State = { ...a };
  for (const [key, value] of Object.entries(b)) {
    if (key === "test_case") {
      merged[key] = { ...(a[key] as State), ...(value as State) };
    } else if (key === "turns" && key in merged) {
      const theirs = value as State[];
      merged[key] = (merged[key] as State[]).map((mine, i) => ({
        ...mine,
        ...theirs[i],
      }));
    } else if (
      key in merged &&
      JSON.stringify(merged[key]) !== JSON.stringify(value)
    ) {
      return undefined;
    } else {
      merged[key] = value;
    }
  }
  return merged;
}

function pack(entries: Entry[]): JevRequest[] {
  const requests: JevRequest[] = [];
  for (const entry of entries) {
    if (!requests.some((request) => request.admit(entry))) {
      requests.push(new JevRequest([entry], entry.state));
    }
  }
  return requests;
}

///////////////////////////////////////////////
// Selection and setup
///////////////////////////////////////////////

function candidates(metrics: AnyMetric[], testCase: unknown): AnyMetric[] {
  let metricType: typeof BaseMetric | typeof BaseConversationalMetric;
  if (testCase instanceof LLMTestCase) metricType = BaseMetric;
  else if (testCase instanceof ConversationalTestCase) {
    metricType = BaseConversationalMetric;
  } else return [];
  // Every whole-metric spec returns `undefined` for a multimodal test case,
  // so none of its metrics could be batched.
  if (testCase.multimodal) return [];
  return metrics.filter(
    (metric) =>
      metric instanceof metricType &&
      effectiveEvalMode(metric) === EvalMode.SYSTEM_ONE,
  );
}

/** Record a failed measure the way `runMetric` does for a metric's own `measure`. */
function fail(
  metric: AnyMetric,
  err: unknown,
  options: SystemOneBatchOptions,
  skipOnMissingParams = options.skipOnMissingParams,
): void {
  if (err instanceof MissingTestCaseParamsError && skipOnMissingParams) {
    metric.skipped = true;
  } else if (options.ignoreErrors) {
    metric.error = (err as Error).message;
  } else {
    throw err;
  }
}

/**
 * Run each candidate's pre-measure setup and build its Jev request. Returns
 * the metrics this batch now owns (including ones whose setup failed) and the
 * entries to send. A metric whose spec declines the test case is left for its
 * own `measure`.
 */
function setup(
  metrics: AnyMetric[],
  testCase: LLMTestCase | ConversationalTestCase,
  options: SystemOneBatchOptions,
): { handled: AnyMetric[]; entries: Entry[] } {
  const handled: AnyMetric[] = [];
  const entries: Entry[] = [];
  for (const metric of metrics) {
    metric.error = undefined;
    metric.skipped = false;
    let prepared;
    try {
      prepareMeasure(metric, testCase);
      const spec = metric.systemOneEvalSpec(testCase);
      prepared = spec && requestFor(metric, testCase, spec);
    } catch (err) {
      fail(metric, err, options);
      handled.push(metric);
      continue;
    }
    if (!prepared) continue;
    startMeasure(metric);
    entries.push({ metric, ...prepared });
    handled.push(metric);
  }
  return { handled, entries };
}

///////////////////////////////////////////////
// Deciding
///////////////////////////////////////////////

/** Each entry's share of a request's cost, by its share of the questions. */
function splitCost(cost: number | null, weights: number[]): (number | null)[] {
  const whole = weights.reduce((sum, weight) => sum + weight, 0);
  return weights.map((weight) =>
    cost === null ? null : (cost * weight) / whole,
  );
}

async function decide(request: JevRequest) {
  // The runner sends the whole test case, the largest state any metric
  // builds, so it checks the budget itself rather than relying on the model
  // to (a custom `DeepEvalBaseSystemOneModel` may not).
  checkContextBudget(request.state, request.questions);
  return request.model.decide(request.state, request.questions);
}

/**
 * Hand each entry its answers and cost share, or the request's error. Nothing
 * falls back to the LLM, as under `system_one` generally.
 */
function settleRequest(
  request: JevRequest,
  result: PromiseSettledResult<Awaited<ReturnType<typeof decide>>>,
  options: SystemOneBatchOptions,
): void {
  if (result.status === "rejected") {
    let err = result.reason;
    if (
      err instanceof SystemOneContextLimitError &&
      request.entries.length === 1
    ) {
      err = contextLimitError(request.entries[0].metric, err);
    }
    for (const entry of request.entries)
      fail(entry.metric, err, options, false);
    return;
  }
  const { answers, cost } = result.value;
  const shares = splitCost(
    cost,
    request.entries.map((entry) => Object.keys(entry.questions).length),
  );
  request.entries.forEach((entry, j) => {
    settle(entry.metric, entry.spec, request.answersFor(answers, j), shares[j]);
  });
}

///////////////////////////////////////////////
// Progress
///////////////////////////////////////////////

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/**
 * One line for the whole batch: a line of the caller's progress display, or a
 * single spinner in place of one per metric. Returns the callback that clears
 * it.
 */
async function showProgress(
  requests: JevRequest[],
  options: SystemOneBatchOptions,
): Promise<() => void> {
  const metrics = requests.flatMap((request) =>
    request.entries.map((entry) => entry.metric),
  );
  if (options.onStart) {
    return options.onStart(
      `    ⚡ Jev: judging ${plural(metrics.length, "metric")} in ` +
        plural(requests.length, "request"),
    );
  }
  if (!metrics.some((metric) => metric.showIndicator)) return () => {};
  const models = [
    ...new Set(requests.map((request) => request.model.getModelName())),
  ].sort();
  const ora = (await import("ora")).default;
  const spinner = ora({
    text:
      `✨ Judging ${metrics.map((metric) => metric.name).join(", ")} with ` +
      `${models.join(", ")} (${plural(requests.length, "request")})`,
    color: "magenta",
    stream: process.stderr,
  }).start();
  return () => spinner.stop();
}

///////////////////////////////////////////////
// Entry point
///////////////////////////////////////////////

/**
 * Judge every batchable metric on `testCase` with Jev, requests running
 * concurrently, and return the metrics it handled (judged, errored or
 * skipped); the caller runs the rest as usual. Returns `[]` when fewer than
 * two metrics qualify.
 */
export async function measureSystemOneBatch(
  metrics: AnyMetric[],
  testCase: LLMTestCase | ConversationalTestCase,
  options: SystemOneBatchOptions,
): Promise<AnyMetric[]> {
  const eligible = candidates(metrics, testCase);
  if (eligible.length < 2) return [];
  const { handled, entries } = setup(eligible, testCase, options);
  const requests = pack(entries);
  if (requests.length === 0) return handled;
  const clear = await showProgress(requests, options);
  try {
    const results = await Promise.allSettled(requests.map(decide));
    requests.forEach((request, i) =>
      settleRequest(request, results[i], options),
    );
  } finally {
    clear();
  }
  return handled;
}
