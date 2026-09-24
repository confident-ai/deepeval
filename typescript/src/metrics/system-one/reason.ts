// Deterministic reasons for a measure Jev decided.
//
// Nothing here calls a model. The reason states who decided, how confident
// the least decisive answer was, and what each answer was in words and
// numbers, so a reader can see exactly where the score came from. The text
// matches the Python SDK's word for word.

import type { BaseMetricCore } from "@/metrics/base-metrics";
import type { QuestionOutcome } from "@/metrics/jev-eval/questions";
import {
  formatG,
  minConfidence,
  verbaliseOutcome,
} from "@/metrics/jev-eval/utils";

function judgeName(metric: BaseMetricCore): string {
  return metric.systemOneModel?.getModelName() ?? "System One";
}

function fmt(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : value.toFixed(2);
}

function outcomeLine(index: number, outcome: QuestionOutcome): string {
  const words = verbaliseOutcome(outcome);
  let numbers: string;
  if (outcome.type === "noul") {
    numbers = `P(yes)=${fmt(outcome.probabilities.true)}`;
  } else if (outcome.type === "score") {
    numbers = `expected level=${fmt(outcome.value)} of 1.00`;
  } else if (!outcome.applicable) {
    numbers = "not applicable";
  } else {
    const values = Object.values(outcome.probabilities);
    numbers = values.length > 0 ? `P=${fmt(Math.max(...values))}` : "P=n/a";
  }
  let line = `${index}. ${outcome.question} -> ${words} (${numbers}`;
  if (outcome.confidence !== null && outcome.confidence !== undefined) {
    line += `, confidence=${fmt(outcome.confidence)}`;
  }
  if (outcome.weight !== 1) line += `, weight=${formatG(outcome.weight)}`;
  line += ")";
  if (outcome.passed !== null && outcome.passed !== undefined) {
    line += ` strict=${outcome.passed ? "pass" : "fail"}`;
  }
  return line;
}

/**
 * The reason for a measure Jev decided: judge, confidence, one line per
 * question, and how the score follows from them.
 */
export function formatSystemOneReason(
  metric: BaseMetricCore,
  outcomes: readonly QuestionOutcome[],
): string {
  const confidence = minConfidence(outcomes);
  let header = `Decided by ${judgeName(metric)}`;
  if (confidence !== null)
    header += `, minimum confidence ${confidence.toFixed(2)}`;
  header += ".";
  const lines = [header, ...outcomes.map((o, i) => outcomeLine(i + 1, o))];

  const applicable = outcomes.filter((o) => o.applicable);
  const score = metric.score;
  if (metric.strictMode) {
    const passed =
      applicable.length > 0 ? applicable.every((o) => o.passed) : true;
    lines.push(
      `Score: ${fmt(score)} (strict mode: ${
        passed
          ? "every applicable question passed"
          : "at least one applicable question failed"
      }).`,
    );
  } else if (applicable.length === 0) {
    lines.push(`Score: ${fmt(score)} (no question applied to this test case).`);
  } else {
    lines.push(
      `Score: ${fmt(score)} (weighted mean of ${applicable.length} ` +
        `applicable question${applicable.length !== 1 ? "s" : ""}).`,
    );
  }
  return lines.join("\n");
}
