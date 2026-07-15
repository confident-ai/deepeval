import { LLMTestCase, SingleTurnParams } from "../../test-case";

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
export function constructGEvalParamsString(
  params: SingleTurnParams[],
): string {
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
export function validateAndSortRubrics(rubrics?: Rubric[]): Rubric[] | undefined {
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
