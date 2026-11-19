// The whole-chain System One path for built-in metrics (`system_one` eval
// mode).
//
// A built-in LLM-as-a-judge metric normally runs a chain: the LLM extracts
// items, a judge decides about each, code turns the decisions into a score and
// the LLM writes a reason. Under `system_one` the metric instead describes
// itself as a `SystemOneEvalSpec`: which test case fields to send as state and
// which bounded questions (the `Noul` / `Score` / `Choice` primitives
// `JevEval` uses) to ask about them. One `decide()` answers them all, the
// score is the weighted mean of the answers, and the reason is deterministic
// text built from the answers. No LLM call is made on this path.
//
// `runSystemOneEval` is what each metric's `measure` calls first. It returns
// `true` when Jev decided the measure (the metric is fully populated and can
// return), and `false` when the mode is not `system_one` or the metric has no
// whole-chain form, so the metric runs its usual chain. Under `system_one`
// nothing falls back: a context overflow throws telling the user to switch
// back to `llm`, any other Jev error surfaces unchanged, and a low-confidence
// result is kept with `metric.confidence` reporting it.

import { z } from "zod";
import { EVAL_MODE_ENV_VAR, EvalMode } from "@/config/eval-mode";
import { DeepEvalError } from "@/errors";
import type { BaseMetricCore } from "@/metrics/base-metrics";
import {
  ConversationalTestCase,
  LLMTestCase,
  MultiTurnParams,
  SingleTurnParams,
} from "@/test-case";
import {
  SystemOneContextLimitError,
  checkContextBudget,
} from "@/models/system-one/limits";
import type {
  SystemOneAnswers,
  SystemOneQuestion,
} from "@/models/system-one/schema";
import {
  Choice,
  Noul,
  Score,
  type JevQuestion,
  type QuestionOutcome,
} from "@/metrics/jev-eval/questions";
import {
  aggregate,
  aggregateStrict,
  buildQuestions,
  constructMultiTurnState,
  constructSingleTurnState,
  formatOutcomesForLogs,
  jsonable,
  markStrict,
  minConfidence,
  outcomesFromAnswers,
} from "@/metrics/jev-eval/utils";
import {
  contextLimitError,
  effectiveEvalMode,
  recordConfidence,
} from "@/metrics/system-one/decision";
import { formatSystemOneReason } from "@/metrics/system-one/reason";
import { constructVerboseLogs } from "@/metrics/utils";

/**
 * What a metric sends to Jev under `system_one` eval mode.
 *
 * `evaluationParams` are the test case fields that become the state (for a
 * `ConversationalTestCase` the turns are always included). `extraState` adds
 * what is not a test case field (a trace, the available tools, a metric
 * option), each under its own top-level key the questions can name.
 * `questions` are the decision points; the score is their weighted mean.
 */
export interface SystemOneEvalSpec {
  evaluationParams: ReadonlyArray<SingleTurnParams | MultiTurnParams>;
  questions: readonly JevQuestion[];
  extraState?: Record<string, unknown>;
}

const TRACE_SPAN_KEYS = [
  "name",
  "type",
  "description",
  "input",
  "output",
  "error",
  "model",
  "retrieval_context",
  "context",
  "expected_output",
  "tools_called",
  "expected_tools",
  "available_tools",
  "agent_handoffs",
] as const;

function camelCase(key: string): string {
  return key.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}

/**
 * A trace as Jev reads it: each span's name, type, inputs, outputs and tool
 * calls, nested under `children`. Token counts, costs, prompts and
 * integration details are dropped; they say nothing about whether the agent
 * did the job and only eat into Jev's context.
 */
export function compactTrace(trace: unknown): unknown {
  if (typeof trace !== "object" || trace === null || Array.isArray(trace)) {
    return trace;
  }
  const source = trace as Record<string, unknown>;
  const span: Record<string, unknown> = {};
  for (const key of TRACE_SPAN_KEYS) {
    // The TS trace dict is camelCase; Jev reads the same keys as in Python.
    const value = source[key] ?? source[camelCase(key)];
    if (
      value !== undefined &&
      value !== null &&
      value !== "" &&
      !(Array.isArray(value) && value.length === 0)
    ) {
      span[key] = value;
    }
  }
  const children = Array.isArray(source.children)
    ? source.children.map(compactTrace)
    : [];
  if (children.length > 0) span.children = children;
  return jsonable(span);
}

const QuestionJson = z.discriminatedUnion("type", [
  z.object({ type: z.literal("noul") }).passthrough(),
  z.object({ type: z.literal("score") }).passthrough(),
  z.object({ type: z.literal("choice") }).passthrough(),
]);

/**
 * Turn a rendered `_experimental_system_one_questions` template (a JSON array
 * of `Noul` / `Score` / `Choice` objects discriminated on `type`) into
 * question objects. Keeping the questions in the template bundle means they
 * can be found, reviewed and overridden like any other prompt.
 */
export function parseQuestions(rendered: string): JevQuestion[] {
  try {
    const items = z.array(QuestionJson).parse(JSON.parse(rendered));
    return items.map((item) => {
      const { type, ...rest } = item as Record<string, any>;
      if (type === "noul") return new Noul(rest as any);
      if (type === "score") return new Score(rest as any);
      return new Choice(rest as any);
    });
  } catch (e) {
    throw new DeepEvalError(
      "_experimental_system_one_questions must render to a JSON array of " +
        `Noul / Score / Choice questions: ${(e as Error).message}`,
    );
  }
}

export interface Prepared {
  spec: SystemOneEvalSpec;
  state: Record<string, unknown>;
  questions: Record<string, SystemOneQuestion>;
}

/**
 * Everything needed for the one `decide()` call, or `undefined` when this
 * measure should not go through Jev at all.
 */
function prepare(
  metric: BaseMetricCore,
  testCase: unknown,
): Prepared | undefined {
  const mode = effectiveEvalMode(metric);
  if (mode !== EvalMode.SYSTEM_ONE) return undefined;
  const spec = metric.systemOneEvalSpec(testCase);
  if (spec === undefined) {
    if (!metric.model) {
      // A wired metric under `system_one` builds no LLM, so there is no other
      // chain to run.
      throw new DeepEvalError(
        `${EVAL_MODE_ENV_VAR}=${mode} runs ${metric.name} on System One, ` +
          `which cannot judge this test case. Switch it back to the LLM with ` +
          `\`evalMode: "${EvalMode.LLM}"\` or ` +
          `\`npx deepeval set-eval-mode ${EvalMode.LLM}\`.`,
      );
    }
    return undefined;
  }
  return requestFor(metric, testCase, spec);
}

/**
 * The state and questions `spec` sends for `testCase`, or `undefined` for a
 * test case type Jev has no state for.
 * @internal Shared with the batched System One path in `evaluate()`.
 */
export function requestFor(
  metric: BaseMetricCore,
  testCase: unknown,
  spec: SystemOneEvalSpec,
): Prepared | undefined {
  const mode = effectiveEvalMode(metric);
  if (!metric.systemOneModel) {
    throw new DeepEvalError(
      `${EVAL_MODE_ENV_VAR}=${mode} runs ${metric.name} on a System One ` +
        `model, but none is configured. Set TYPESAFE_API_KEY or switch back ` +
        `with \`npx deepeval set-eval-mode ${EvalMode.LLM}\`.`,
    );
  }

  let state: Record<string, unknown>;
  if (testCase instanceof ConversationalTestCase) {
    const params = spec.evaluationParams.filter((p) =>
      (Object.values(MultiTurnParams) as string[]).includes(p),
    ) as MultiTurnParams[];
    state = constructMultiTurnState(params, testCase);
  } else if (testCase instanceof LLMTestCase) {
    const params = spec.evaluationParams.filter((p) =>
      (Object.values(SingleTurnParams) as string[]).includes(p),
    ) as SingleTurnParams[];
    state = constructSingleTurnState(params, testCase);
    if (Object.keys(state.test_case as object).length === 0) {
      delete state.test_case;
    }
  } else {
    return undefined;
  }
  for (const [key, value] of Object.entries(spec.extraState ?? {})) {
    if (value !== undefined && value !== null) state[key] = jsonable(value);
  }
  return { spec, state, questions: buildQuestions(spec.questions) };
}

function populate(
  metric: BaseMetricCore,
  spec: SystemOneEvalSpec,
  outcomesIn: QuestionOutcome[],
): void {
  let outcomes = outcomesIn;
  const confidence = minConfidence(outcomes);
  if (metric.strictMode) {
    outcomes = markStrict(spec.questions, outcomes);
    metric.score = aggregateStrict(outcomes);
  } else {
    metric.score = aggregate(outcomes);
  }
  metric.systemOneOutcomes = outcomes;
  metric.scoreBreakdown = outcomes.map((o) => ({ ...o }));
  metric.confidence = confidence;
  metric.reason = metric.includeReason
    ? formatSystemOneReason(metric, outcomes)
    : undefined;
  metric.success = metric.isSuccessful();
  metric.evaluationModel = metric.systemOneModel!.getModelName();
  metric.verboseLogs = constructVerboseLogs(metric, [
    `Decided by System One (${metric.evaluationModel}); no LLM call was made.`,
    `Questions:\n${formatOutcomesForLogs(outcomes)}`,
    `Score: ${metric.score}\nConfidence: ${confidence}\nReason: ${metric.reason}`,
  ]);
}

/** Decide the whole measure with Jev. See the module comment. */
export async function runSystemOneEval(
  metric: BaseMetricCore,
  testCase: unknown,
): Promise<boolean> {
  const prepared = prepare(metric, testCase);
  if (prepared === undefined) return false;
  const { spec, state, questions } = prepared;
  let decision;
  // The runner sends the whole test case, the largest state any metric
  // builds, so it checks the budget itself rather than relying on the model
  // to (a custom `DeepEvalBaseSystemOneModel` may not).
  try {
    checkContextBudget(state, questions);
    decision = await metric.systemOneModel!.decide(state, questions);
  } catch (err) {
    if (err instanceof SystemOneContextLimitError) {
      throw contextLimitError(metric, err);
    }
    throw err;
  }
  settle(metric, spec, decision.answers, decision.cost);
  return true;
}

/**
 * Book Jev's answers and cost for one measure and fill in every field it
 * reports.
 * @internal Shared with the batched System One path in `evaluate()`.
 */
export function settle(
  metric: BaseMetricCore,
  spec: SystemOneEvalSpec,
  answers: SystemOneAnswers,
  cost: number | null,
): void {
  metric.accrueCost(cost);
  recordConfidence(metric, answers);
  populate(metric, spec, outcomesFromAnswers(spec.questions, answers));
}
