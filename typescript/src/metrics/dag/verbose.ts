import { prettifyList } from "@/metrics/utils";
import { GEval } from "@/metrics/g-eval/g-eval";
import { ConversationalGEval } from "@/metrics/conversational-g-eval/conversational-g-eval";
import type { JudgementVerdict } from "@/metrics/dag/schema";
import type { ChildMetric } from "@/metrics/dag/types";

export function taskVerboseLog(
  underscores: string,
  stars: string,
  nodeType: string,
  label: string | undefined,
  instructions: string,
  outputLabel: string,
  depth: number,
  output: unknown,
): string {
  return (
    `${underscores}\n` +
    `| ${nodeType} | Level == ${depth} |\n` +
    `${stars}\n` +
    `Label: ${label ?? "None"}\n\n` +
    "Instructions:\n" +
    `${instructions}\n\n` +
    `${outputLabel}:\n${output}\n`
  );
}

export function judgementVerboseLog(
  nodeType: string,
  underscores: number,
  stars: number,
  label: string | undefined,
  criteria: string,
  depth: number,
  verdict: JudgementVerdict,
): string {
  return (
    `${"_".repeat(underscores)}\n` +
    `| ${nodeType} | Level == ${depth} |\n` +
    `${"*".repeat(stars)}\n` +
    `Label: ${label ?? "None"}\n\n` +
    "Criteria:\n" +
    `${criteria}\n\n` +
    `Verdict: ${verdict.verdict}\n` +
    `Reason: ${verdict.reason}\n`
  );
}

export function verdictVerboseLog(
  underscores: string,
  stars: string,
  nodeType: string,
  verdict: string | boolean,
  depth: number,
  childMetric?: ChildMetric,
): string {
  const type = childMetric ? `${childMetric.name} Metric` : "Deterministic";
  let log =
    `${underscores}\n` +
    `| ${nodeType} | Level == ${depth} |\n` +
    `${stars}\n` +
    `Verdict: ${verdict}\n` +
    `Type: ${type}`;
  if (
    childMetric instanceof GEval ||
    childMetric instanceof ConversationalGEval
  ) {
    log += `\n\nCriteria:\n${childMetric.criteria}\n`;
    log += `Evaluation Steps:\n${prettifyList(childMetric.evaluationSteps ?? [])}`;
  } else if (childMetric) {
    log += `\n\n${childMetric.verboseLogs}`;
  }
  return log;
}
