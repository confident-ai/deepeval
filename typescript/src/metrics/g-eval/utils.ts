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
 * Start and length of the last run of tokens spelling `rawScore`. The reason
 * may repeat its digits, and some tokenizers split "10" into "1", "0".
 */
function findScoreSpan(
  logProbs: ContentTokenLogProbs[],
  rawScore: number,
): [number, number] | undefined {
  const scoreStr = String(rawScore);
  for (let start = logProbs.length - 1; start >= 0; start--) {
    for (let length = scoreStr.length; length >= 1; length--) {
      const span = logProbs.slice(start, start + length);
      const spelled = span.map((entry) => entry.token).join("");
      if (span.length === length && spelled === scoreStr) {
        return [start, length];
      }
    }
  }
  return undefined;
}

/** Whether `token` could be the first digit of a larger score, like "1" of 10. */
function isPrefixOfLargerScore(token: string, scoreMax: number): boolean {
  for (let score = parseInt(token, 10) + 1; score <= scoreMax; score++) {
    if (String(score).startsWith(token)) return true;
  }
  return false;
}

/**
 * Average the scores the judge considered, weighted by their probability — the
 * G-Eval paper's scoring, and a port of Python's `calculate_weighted_summed_score`.
 * Candidates outside `scoreMin`..`scoreMax` are ignored, so the result stays
 * inside the score range. Falls back to `rawScore` when there is nothing to
 * weight.
 */
export function calculateWeightedSummedScore(
  rawScore: number,
  logProbs: ContentTokenLogProbs[] | undefined,
  scoreMax = 10,
  scoreMin = 0,
): number {
  const span = logProbs && findScoreSpan(logProbs, rawScore);
  if (!logProbs || !span) return rawScore;
  const [start, length] = span;
  const scoreToken = logProbs[start];

  let weightedSum = 0;
  let totalProb = 0;
  for (const candidate of scoreToken.topLogProbs) {
    if (candidate.logprob < MIN_LOG_PROB || !/^\d+$/.test(candidate.token)) {
      continue;
    }
    if (parseInt(candidate.token, 10) > scoreMax) continue;

    let score: number;
    let logprob = candidate.logprob;
    if (candidate.token === scoreToken.token) {
      // The emitted score, including the rest of its span.
      score = rawScore;
      for (const rest of logProbs.slice(start + 1, start + length)) {
        logprob += rest.logprob;
      }
    } else if (isPrefixOfLargerScore(candidate.token, scoreMax)) {
      continue;
    } else {
      score = parseInt(candidate.token, 10);
      // e.g. "0" under a 1-5 rubric would pull the score below 1
      if (score < scoreMin) continue;
    }

    const prob = Math.exp(logprob);
    weightedSum += score * prob;
    totalProb += prob;
  }

  return totalProb > 0 ? weightedSum / totalProb : rawScore;
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
  options: {
    topLogprobs: number;
    strictMode: boolean;
    scoreMax?: number;
    scoreMin?: number;
  },
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
          : calculateWeightedSummedScore(
              score,
              logProbs,
              options.scoreMax,
              options.scoreMin,
            ),
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
