import { LLMTestCase, SingleTurnParams } from "@/test-case";
import type { ContentTokenLogProbs } from "@/models";
import { extractJson } from "@/models/utils";
import type { BaseMetricCore } from "@/metrics/base-metrics";
import { generateWithSchema } from "@/metrics/utils";
import { ReasonScoreSchema } from "@/metrics/g-eval/schema";

/** A score band and the outcome it represents (GEval rubric). */
export interface Rubric {
  scoreRange: [number, number];
  expectedOutcome: string;
}

/** Display label for each evaluation param (mirrors Python's G_EVAL_PARAMS). */
export const G_EVAL_PARAMS: Partial<Record<SingleTurnParams, string>> = {
  [SingleTurnParams.INPUT]: "Input",
  [SingleTurnParams.ACTUAL_OUTPUT]: "Actual Output",
  [SingleTurnParams.EXPECTED_OUTPUT]: "Expected Output",
  [SingleTurnParams.CONTEXT]: "Context",
  [SingleTurnParams.RETRIEVAL_CONTEXT]: "Retrieval Context",
  [SingleTurnParams.TOOLS_CALLED]: "Tools Called",
  [SingleTurnParams.EXPECTED_TOOLS]: "Expected Tools",
};

function getParamValue(
  testCase: LLMTestCase,
  param: SingleTurnParams,
): unknown {
  switch (param) {
    case SingleTurnParams.INPUT:
      return testCase.input;
    case SingleTurnParams.ACTUAL_OUTPUT:
      return testCase.actualOutput;
    case SingleTurnParams.EXPECTED_OUTPUT:
      return testCase.expectedOutput;
    case SingleTurnParams.CONTEXT:
      return testCase.context;
    case SingleTurnParams.RETRIEVAL_CONTEXT:
      return testCase.retrievalContext;
    case SingleTurnParams.TOOLS_CALLED:
      return testCase.toolsCalled;
    case SingleTurnParams.EXPECTED_TOOLS:
      return testCase.expectedTools;
    default:
      return undefined;
  }
}

/** Format a param's value as Python's `f"{value}"` would (text-only). */
function formatValue(value: unknown): string {
  if (value == null) return "None";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return (
      "[" +
      value
        .map((v) => (typeof v === "string" ? `'${v}'` : String(v)))
        .join(", ") +
      "]"
    );
  }
  return String(value);
}

/** Join param labels: "A", "A and B", or "A, B, and C". */
export function constructGEvalParamsString(params: SingleTurnParams[]): string {
  const labels = params.map((p) => G_EVAL_PARAMS[p] ?? p);
  if (labels.length === 1) return labels[0];
  if (labels.length === 2) return labels.join(" and ");
  return `${labels.slice(0, -1).join(", ")}, and ${labels[labels.length - 1]}`;
}

/** Render the relevant test-case fields as text for the eval prompt. */
export function constructTestCaseString(
  params: SingleTurnParams[],
  testCase: LLMTestCase,
): string {
  let text = "";
  for (const param of params) {
    const value = getParamValue(testCase, param);
    text += `${G_EVAL_PARAMS[param] ?? param}:\n${formatValue(value)} \n\n`;
  }
  return text;
}

/** "1. step\n2. step\n…" */
export function numberEvaluationSteps(steps: string[]): string {
  return steps.map((s, i) => `${i + 1}. ${s}\n`).join("");
}

export function formatRubrics(rubrics?: Rubric[]): string | null {
  if (!rubrics) return null;
  return rubrics
    .map(({ scoreRange: [start, end], expectedOutcome }) =>
      start === end
        ? `${start}: ${expectedOutcome}`
        : `${start}-${end}: ${expectedOutcome}`,
    )
    .join("\n");
}

export function getScoreRange(rubrics?: Rubric[]): [number, number] {
  if (!rubrics) return [0, 10];
  return [rubrics[0].scoreRange[0], rubrics[rubrics.length - 1].scoreRange[1]];
}

/** Sort rubrics by start and reject overlaps. Returns undefined for none. */
export function validateAndSortRubrics(
  rubrics?: Rubric[],
): Rubric[] | undefined {
  if (!rubrics || rubrics.length === 0) return undefined;
  const sorted = [...rubrics].sort((a, b) => a.scoreRange[0] - b.scoreRange[0]);
  for (let i = 0; i < sorted.length; i++) {
    const [, aEnd] = sorted[i].scoreRange;
    for (let j = i + 1; j < sorted.length; j++) {
      const [bStart] = sorted[j].scoreRange;
      if (aEnd >= bStart) {
        throw new Error(
          `Overlapping score ranges: ${JSON.stringify(sorted[i].scoreRange)} and ${JSON.stringify(sorted[j].scoreRange)}`,
        );
      }
    }
  }
  return sorted;
}

/** Below this linear probability a candidate token is noise (Python's 1% floor). */
const MIN_LOG_PROB = Math.log(0.01);

/**
 * Refine a discrete score into the expectation over the score tokens the model
 * considered, weighted by their probabilities — the G-Eval paper's scoring, and
 * a port of Python's `calculate_weighted_summed_score`.
 *
 * Tokenizer-aware (#3517): the score's full token span is located from the
 * end (the reasoning may repeat digits first), so a judge that splits numbers
 * into single digits — Qwen emits 10 as "1","0" — is weighted over that span.
 * A candidate that is a strict prefix of another valid score, like "1" under
 * a 0–10 rubric, cannot be resolved from this response (only the branch the
 * model took is recorded), so it is dropped rather than guessed at.
 *
 * Returns `rawScore` unchanged when the log-probs can't support a better
 * answer: no matching span, or every candidate filtered out.
 */
export function calculateWeightedSummedScore(
  rawScore: number,
  logProbs: ContentTokenLogProbs[] | undefined,
  scoreMax = 10,
): number {
  if (!logProbs?.length) return rawScore;

  const scoreStr = String(rawScore);
  // Scan from the end: the reasoning may spell the score earlier than the
  // model's final answer.
  let spanStart = -1;
  let spanLen = 0;
  for (let i = logProbs.length - 1; i >= 0 && spanStart < 0; i--) {
    for (let k = scoreStr.length; k >= 1; k--) {
      if (i + k > logProbs.length) continue;
      const joined = logProbs
        .slice(i, i + k)
        .map((entry) => entry.token)
        .join("");
      if (joined === scoreStr) {
        spanStart = i;
        spanLen = k;
        break;
      }
    }
  }
  if (spanStart < 0) return rawScore;
  const first = logProbs[spanStart];

  const isAmbiguousPrefix = (token: string): boolean => {
    const value = parseInt(token, 10);
    for (let s = value + 1; s <= scoreMax; s++) {
      if (String(s).startsWith(token)) return true;
    }
    return false;
  };

  const linearProbByScore = new Map<number, number>();
  let sumLinearProb = 0;

  for (const candidate of first.topLogProbs) {
    if (candidate.logprob < MIN_LOG_PROB) continue;
    // Anything non-decimal isn't a score, and would parse to NaN.
    if (!/^\d+$/.test(candidate.token)) continue;

    let tokenScore = parseInt(candidate.token, 10);
    // A candidate outside the rubric is not a score.
    if (tokenScore > scoreMax) continue;

    let linearProb = Math.exp(candidate.logprob);
    if (candidate.token === first.token) {
      // The branch the model actually took: it spells the raw score over one
      // token or the whole span, and each later emitted token's own logprob
      // is the conditional that branch needs.
      for (let j = 1; j < spanLen; j++) {
        linearProb *= Math.exp(logProbs[spanStart + j].logprob);
      }
      tokenScore = rawScore;
    } else if (isAmbiguousPrefix(candidate.token)) {
      continue;
    }
    linearProbByScore.set(
      tokenScore,
      (linearProbByScore.get(tokenScore) ?? 0) + linearProb,
    );
    sumLinearProb += linearProb;
  }

  if (sumLinearProb === 0) return rawScore;

  let sumWeightedScores = 0;
  for (const [score, prob] of linearProbByScore) {
    sumWeightedScores += score * prob;
  }
  return sumWeightedScores / sumLinearProb;
}

/**
 * Score a G-Eval prompt, preferring the log-prob path so the score is a
 * probability-weighted expectation rather than the single integer the model
 * happened to emit. Shared by `GEval` and `ConversationalGEval`.
 *
 * Falls back to the plain structured call whenever the raw path is unavailable
 * or fails for any reason — a provider can advertise log-probs and still reject
 * the parameter, and Python likewise treats a missing raw path as routine.
 */
export async function evaluateGEvalPrompt(
  metric: BaseMetricCore,
  prompt: string,
  options: { topLogprobs: number; strictMode: boolean },
): Promise<[number, string]> {
  const model = metric.model;
  if (model?.generateRaw && model.supportsLogProbs() !== false) {
    try {
      const { output, cost, logProbs } = await model.generateRaw(prompt, {
        topLogprobs: options.topLogprobs,
      });
      metric.accrueCost(cost);
      const { score, reason } = ReasonScoreSchema.parse(extractJson(output));
      return [
        options.strictMode
          ? score
          : calculateWeightedSummedScore(score, logProbs),
        reason,
      ];
    } catch {
      // Fall through to the structured path below.
    }
  }

  const { score, reason } = await generateWithSchema(
    metric,
    prompt,
    ReasonScoreSchema,
  );
  return [score, reason];
}

export function validateCriteriaAndEvaluationSteps(
  criteria?: string,
  evaluationSteps?: string[],
): void {
  if (criteria == null && evaluationSteps == null) {
    throw new Error("Either 'criteria' or 'evaluationSteps' must be provided.");
  }
  if (criteria != null && criteria.trim() === "") {
    throw new Error("Criteria provided cannot be an empty string.");
  }
  if (evaluationSteps != null && evaluationSteps.length === 0) {
    throw new Error(
      "'evaluationSteps' must not be an empty list. Either omit evaluation steps or include a non-empty list of steps.",
    );
  }
}
