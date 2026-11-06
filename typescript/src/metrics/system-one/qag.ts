// System One (Jev) verdicts for QAG metrics.
//
// Under the `hybrid` eval mode the decision step of a QAG metric is a set of
// Noul questions, one per item, answered by a System One model. The LLM still
// extracts the items and writes the reasons. P(yes) is thresholded into the
// metric's verdict vocabulary. A Jev call that fails at runtime hands the
// verdicts to the LLM instead (see decision.ts).

import { DeepEvalError } from "@/errors";
import type { BaseMetricCore } from "@/metrics/base-metrics";
import type { NoulQuestion } from "@/models/system-one/schema";
import { jsonable } from "@/metrics/jev-eval/utils";
import {
  SYSTEM_ONE_YES_THRESHOLD,
  recordConfidence,
  systemOneActive,
  systemOneCall,
} from "@/metrics/system-one/decision";

export const SYSTEM_ONE_BORDERLINE_LOW = 0.35;
export const SYSTEM_ONE_BORDERLINE_HIGH = 0.65;

const SENTENCE_END = /(?<=[.!?])\s+/;

/**
 * Split text into sentences for metrics whose LLM prompt both splits and
 * judges in one step; under `hybrid` the split is done here so Jev can judge
 * each sentence.
 */
export function splitSentences(text: string | undefined | null): string[] {
  return (text ?? "")
    .split(SENTENCE_END)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/** `"yes"`, `"no"`, or the metric's own borderline token (e.g. `"idk"`). */
export type SystemOneVerdict = string;

export interface SystemOneVerdictSpec<Item = unknown, V = any> {
  /** The question asked about each item. */
  instructions: string;
  items: readonly Item[];
  /** The item's name in the question and in the state (`${itemKey}s`). */
  itemKey: string;
  /** Shared context every question can refer to (e.g. `{ input }`). */
  state?: Record<string, unknown>;
  criteria?: [unknown, unknown];
  /**
   * The metric's borderline token (`"idk"`). When set, P(yes) in
   * [0.35, 0.65] maps to it; otherwise the verdict is yes/no at 0.5.
   */
  borderline?: string;
  /** Shape the metric's own verdict object; defaults to `{ verdict, reason }`. */
  buildVerdict?: (item: Item, verdict: SystemOneVerdict, p: number) => V;
}

export function verdictFromProbability(
  probability: number,
  borderline?: string,
): SystemOneVerdict {
  if (borderline !== undefined) {
    if (probability > SYSTEM_ONE_BORDERLINE_HIGH) return "yes";
    if (probability < SYSTEM_ONE_BORDERLINE_LOW) return "no";
    return borderline;
  }
  return probability >= SYSTEM_ONE_YES_THRESHOLD ? "yes" : "no";
}

function questionKey(spec: SystemOneVerdictSpec<any, any>, i: number): string {
  return `${spec.itemKey}_${i}`;
}

function request(spec: SystemOneVerdictSpec<any, any>): {
  state: Record<string, unknown>;
  questions: Record<string, NoulQuestion>;
} {
  const items = spec.items.map(jsonable);
  const state = {
    ...(jsonable(spec.state ?? {}) as Record<string, unknown>),
    [`${spec.itemKey}s`]: items,
  };
  const questions: Record<string, NoulQuestion> = {};
  items.forEach((item, i) => {
    questions[questionKey(spec, i)] = {
      type: "noul",
      instructions: { [spec.itemKey]: item, question: spec.instructions },
      true: spec.criteria?.[0],
      false: spec.criteria?.[1],
    };
  });
  return { state, questions };
}

/**
 * Jev's verdict per item, or `undefined` when the LLM decides (not active, or
 * the call failed under `hybrid`).
 */
export async function systemOneVerdicts<Item, V>(
  metric: BaseMetricCore,
  spec: SystemOneVerdictSpec<Item, V> | undefined,
): Promise<V[] | undefined> {
  if (!systemOneActive(metric, spec)) return undefined;
  const s = spec as SystemOneVerdictSpec<Item, V>;
  if (s.items.length === 0) return [];
  const { state, questions } = request(s);
  const result = await systemOneCall(metric, () =>
    metric.systemOneModel!.noul(state, questions),
  );
  if (result === undefined) return undefined;
  metric.accrueCost(result.cost);
  recordConfidence(metric, result.answers);
  return s.items.map((item, i) => {
    const answer = result.answers[questionKey(s, i)];
    if (!answer) {
      throw new DeepEvalError(
        `System One model returned no answer for ${s.itemKey} ${i}.`,
      );
    }
    const p = answer.probability;
    const verdict = verdictFromProbability(p, s.borderline);
    return s.buildVerdict
      ? s.buildVerdict(item, verdict, p)
      : ({ verdict, reason: `P(yes)=${p.toFixed(2)}` } as V);
  });
}

/**
 * THE QAG entry point: Jev's per-item verdicts when the eval mode uses
 * System One, else (or when that call fails under `hybrid`) the LLM's.
 */
export async function generateQagVerdicts<V>(
  metric: BaseMetricCore,
  options: {
    systemOne?: SystemOneVerdictSpec<any, V>;
    llm: () => Promise<V[]>;
  },
): Promise<V[]> {
  return (
    (await systemOneVerdicts(metric, options.systemOne)) ??
    (await options.llm())
  );
}
