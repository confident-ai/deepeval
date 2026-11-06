// System One (Jev) plumbing shared by every metric's decision points.
//
// Three eval modes (`@/config/eval-mode`):
//
// - `llm`: `systemOneActive` is false, every helper returns `undefined` and
//   the metric takes its LLM branch.
// - `hybrid`: the helpers ask Jev. The LLM is already part of the chain, so a
//   Jev call that fails at runtime hands that one decision to the LLM and
//   records why on `metric.systemOneFallbackReason`.
// - `system_one`: the whole-chain runner (`runner.ts`) asks Jev once per
//   measure and nothing falls back. Metrics with no whole-chain form run as
//   `hybrid` (`effectiveEvalMode`).
//
// Every helper here returns `undefined` for "the LLM decides this one", so a
// metric reads `(await systemOneX(this, spec)) ?? (await llmX())`.

import {
  EVAL_MODE_ENV_VAR,
  EvalMode,
  resolveEvalMode,
  usesSystemOne,
  type EvalModeName,
} from "@/config/eval-mode";
import { DeepEvalError } from "@/errors";
import { BaseMetricCore } from "@/metrics/base-metrics";
import { SystemOneContextLimitError } from "@/models/system-one/limits";
import {
  SystemOneAnswers,
  type ChoiceAnswer,
  type NoulAnswer,
  type ScoreAnswer,
} from "@/models/system-one/schema";
import { jsonable } from "@/metrics/jev-eval/utils";

export const SYSTEM_ONE_YES_THRESHOLD = 0.5;

/**
 * Whether the metric can run as one Jev request (it overrides
 * `systemOneEvalSpec`).
 */
export function hasWholeMetricForm(metric: BaseMetricCore): boolean {
  return (
    metric.systemOneEvalSpec !== BaseMetricCore.prototype.systemOneEvalSpec
  );
}

/**
 * The mode the metric actually runs in. A metric with no whole-metric form
 * (DAG, whose task nodes need the LLM, or a user's own subclass) cannot
 * honour `system_one`, so it runs as `hybrid`: Jev at its decision points,
 * the LLM covering a failed call.
 */
export function effectiveEvalMode(metric: BaseMetricCore): EvalModeName {
  const mode = metric.evalMode ?? resolveEvalMode();
  if (mode === EvalMode.SYSTEM_ONE && !hasWholeMetricForm(metric)) {
    return EvalMode.HYBRID;
  }
  return mode;
}

export function systemOneActive(
  metric: BaseMetricCore,
  spec: unknown,
): boolean {
  if (spec === undefined || spec === null) return false;
  const mode = effectiveEvalMode(metric);
  if (!usesSystemOne(mode)) return false;
  if (metric._systemOneDisabled) return false;
  if (!metric.systemOneModel) {
    throw new DeepEvalError(
      `${EVAL_MODE_ENV_VAR}=${mode} routes decisions to a System One model, ` +
        `but ${metric.name} has none configured. Set TYPESAFE_API_KEY or ` +
        `switch back with \`npx deepeval set-eval-mode ${EvalMode.LLM}\`.`,
    );
  }
  return true;
}

///////////////////////////////////////////////
// Bookkeeping
///////////////////////////////////////////////

type AnswerMap = Record<string, { confidence: number }>;

/**
 * Fold the least decisive answer of one Jev request into `metric.confidence`
 * (minimum across the measure).
 */
export function recordConfidence(
  metric: BaseMetricCore,
  answers: SystemOneAnswers | AnswerMap,
): number | undefined {
  let confidence: number | undefined;
  if (answers instanceof SystemOneAnswers) {
    confidence = answers.minConfidence();
  } else {
    const values = Object.values(answers).map((a) => a.confidence);
    confidence = values.length > 0 ? Math.min(...values) : undefined;
  }
  if (confidence === undefined) return undefined;
  const current = metric.confidence;
  metric.confidence =
    current === undefined || current === null
      ? confidence
      : Math.min(current, confidence);
  return confidence;
}

function noteFallback(metric: BaseMetricCore, reason: string): void {
  const previous = metric.systemOneFallbackReason;
  metric.systemOneFallbackReason = previous ? `${previous}; ${reason}` : reason;
}

///////////////////////////////////////////////
// Failures
///////////////////////////////////////////////

const NETWORK_CODES = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "ETIMEDOUT",
  "ENOTFOUND",
  "EAI_AGAIN",
  "EPIPE",
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_SOCKET",
]);

function statusOf(err: any): number | undefined {
  const status = err?.status ?? err?.statusCode ?? err?.response?.status;
  return typeof status === "number" ? status : undefined;
}

/**
 * Whether an error came from asking Jev (SDK, network, budget, missing SDK or
 * key) rather than from a bug. Programming errors always propagate.
 */
function isSystemOneError(err: unknown): boolean {
  if (err instanceof DeepEvalError) return true;
  if (!(err instanceof Error)) return false;
  const e = err as any;
  if (statusOf(e) !== undefined) return true;
  if (typeof e.code === "string" && NETWORK_CODES.has(e.code)) return true;
  if (/TypeSafe|APIError|Connection|Timeout|Abort|RateLimit/i.test(err.name)) {
    return true;
  }
  if (err.message.includes("@typesafe-ai/sdk")) return true;
  // Node's fetch reports network failures as `TypeError: fetch failed`.
  return err instanceof TypeError && err.message === "fetch failed";
}

/** Failures that will not get better on the next Jev call in this measure. */
function isPermanent(err: unknown): boolean {
  const status = statusOf(err);
  if (status === 401 || status === 403) return true;
  if (!(err instanceof Error)) return false;
  if (/Authentication|PermissionDenied/i.test(err.name)) return true;
  const text = err.message.toLowerCase();
  return text.includes("api key") || text.includes("@typesafe-ai/sdk");
}

function describeFailure(err: unknown): string {
  if (err instanceof SystemOneContextLimitError) {
    return `context limit (${err.estimatedTokens} est. tokens > ${err.limitTokens})`;
  }
  if (err instanceof Error) {
    if (/Timeout|Abort/i.test(err.name)) return `timeout (${err.message})`;
    return `${err.name}: ${err.message}`;
  }
  return String(err);
}

/**
 * The error a `system_one` metric throws when its test case does not fit in
 * Jev's context: Jev cannot judge it, and the LLM can.
 */
export function contextLimitError(
  metric: BaseMetricCore,
  err: SystemOneContextLimitError,
): DeepEvalError {
  return new DeepEvalError(
    `${metric.name} could not run on System One: ${err.message} This test ` +
      `case is too large for Jev to judge in one request. Switch this metric ` +
      `back to the LLM with \`evalMode: "${EvalMode.LLM}"\` or ` +
      `\`npx deepeval set-eval-mode ${EvalMode.LLM}\`.`,
  );
}

/**
 * Decide what a failed Jev call means for this metric. Under `hybrid` the
 * failure is recorded on `metric.systemOneFallbackReason` and swallowed so the
 * caller takes its LLM branch for that decision; permanent failures (auth)
 * also keep the rest of the measure on the LLM. In every other mode it
 * throws: a context overflow as the switch-back-to-`llm` error, anything else
 * unchanged. Programming errors always propagate.
 */
export function handleSystemOneFailure(
  metric: BaseMetricCore,
  err: unknown,
): void {
  if (!isSystemOneError(err)) throw err;
  if (effectiveEvalMode(metric) === EvalMode.HYBRID) {
    noteFallback(metric, describeFailure(err));
    if (isPermanent(err)) metric._systemOneDisabled = true;
    return;
  }
  if (err instanceof SystemOneContextLimitError) {
    throw contextLimitError(metric, err);
  }
  throw err;
}

/**
 * Run one System One model call. Returns its result, or `undefined` when the
 * call failed and the metric (`hybrid` only) takes its LLM branch.
 */
export async function systemOneCall<T>(
  metric: BaseMetricCore,
  fn: () => Promise<T>,
): Promise<T | undefined> {
  try {
    return await fn();
  } catch (err) {
    handleSystemOneFailure(metric, err);
    return undefined;
  }
}

///////////////////////////////////////////////
// Single decisions
///////////////////////////////////////////////

export interface SystemOneBinarySpec {
  instructions: unknown;
  state: Record<string, unknown>;
  criteria?: [unknown, unknown];
}

export interface SystemOneChoiceSpec {
  instructions: unknown;
  options: readonly string[];
  state: Record<string, unknown>;
}

export interface SystemOneScoreSpec {
  instructions: unknown;
  levels: readonly string[];
  state: Record<string, unknown>;
}

async function askOne<A extends { confidence: number }>(
  metric: BaseMetricCore,
  ask: () => Promise<{ answers: Record<string, A>; cost: number | null }>,
): Promise<A | undefined> {
  const result = await systemOneCall(metric, ask);
  if (result === undefined) return undefined;
  metric.accrueCost(result.cost);
  recordConfidence(metric, result.answers);
  const answer = result.answers.verdict;
  if (!answer) {
    throw new DeepEvalError("System One model returned no answer for verdict.");
  }
  return answer;
}

/** Jev's P(yes) for one yes/no decision, or `undefined` for the LLM. */
export async function systemOneProbability(
  metric: BaseMetricCore,
  spec: SystemOneBinarySpec | undefined,
): Promise<number | undefined> {
  if (!systemOneActive(metric, spec)) return undefined;
  const s = spec as SystemOneBinarySpec;
  const answer = await askOne<NoulAnswer>(metric, () =>
    metric.systemOneModel!.noul(jsonable(s.state), {
      verdict: {
        type: "noul",
        instructions: s.instructions,
        true: s.criteria?.[0],
        false: s.criteria?.[1],
      },
    }),
  );
  return answer?.probability;
}

/** Jev's pick among `spec.options`, or `undefined` for the LLM. */
export async function systemOneChoice(
  metric: BaseMetricCore,
  spec: SystemOneChoiceSpec | undefined,
): Promise<ChoiceAnswer | undefined> {
  if (!systemOneActive(metric, spec)) return undefined;
  const s = spec as SystemOneChoiceSpec;
  const options: Record<string, null> = {};
  for (const option of s.options) options[option] = null;
  return askOne<ChoiceAnswer>(metric, () =>
    metric.systemOneModel!.choice(jsonable(s.state), {
      verdict: { type: "choice", instructions: s.instructions, options },
    }),
  );
}

/**
 * Jev's score on `spec.levels`, mapped onto [0, 1], or `undefined` for the
 * LLM.
 */
export async function systemOneScore(
  metric: BaseMetricCore,
  spec: SystemOneScoreSpec | undefined,
): Promise<number | undefined> {
  if (!systemOneActive(metric, spec)) return undefined;
  const s = spec as SystemOneScoreSpec;
  const answer = await askOne<ScoreAnswer>(metric, () =>
    metric.systemOneModel!.score(jsonable(s.state), {
      verdict: {
        type: "score",
        instructions: s.instructions,
        levels: [...s.levels],
      },
    }),
  );
  return answer?.normalized;
}

///////////////////////////////////////////////
// Judgements with an LLM fallback
///////////////////////////////////////////////

/**
 * A yes/no judgement (a DAG binary node): Jev when active, else the LLM.
 * The Jev result reads `{ verdict, reason: "P(true)=0.93" }`.
 */
export async function generateBinaryJudgement<
  T extends { verdict: boolean; reason?: string | null },
>(
  metric: BaseMetricCore,
  options: { systemOne?: SystemOneBinarySpec; llm: () => Promise<T> },
): Promise<T> {
  const p = await systemOneProbability(metric, options.systemOne);
  if (p !== undefined) {
    return {
      verdict: p >= SYSTEM_ONE_YES_THRESHOLD,
      reason: `P(true)=${p.toFixed(2)}`,
    } as T;
  }
  return options.llm();
}

/**
 * A pick among closed options (a DAG non-binary node): Jev when active, else
 * the LLM.
 */
export async function generateChoiceJudgement<
  T extends { verdict: string; reason?: string | null },
>(
  metric: BaseMetricCore,
  options: { systemOne?: SystemOneChoiceSpec; llm: () => Promise<T> },
): Promise<T> {
  const answer = await systemOneChoice(metric, options.systemOne);
  if (answer !== undefined) {
    return {
      verdict: answer.choice,
      reason:
        `P=${(answer.probabilities[answer.choice] ?? 0).toFixed(2)}, ` +
        `confidence=${answer.confidence.toFixed(2)}`,
    } as T;
  }
  return options.llm();
}

/**
 * The reason that stands in for the LLM's when Jev made a decision the LLM
 * would have explained in the same response.
 */
export function formatDecisionReason(
  metric: BaseMetricCore,
  what: string,
  value: number,
): string {
  const name = metric.systemOneModel?.getModelName() ?? "System One";
  let text = `Decided by ${name}: ${what} ${value.toFixed(2)}`;
  if (metric.confidence !== undefined && metric.confidence !== null) {
    text += `, confidence ${metric.confidence.toFixed(2)}`;
  }
  return `${text}.`;
}
