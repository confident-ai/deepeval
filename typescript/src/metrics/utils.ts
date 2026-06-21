import type { ZodType } from "zod";
import { DeepEvalBaseLLM, OpenAIModel } from "../models";
import {
  LLMTestCase,
  SingleTurnParams,
  ToolCall,
  ArenaTestCase,
} from "../test-case";
import { DeepEvalError, MissingTestCaseParamsError } from "../errors";
import { BaseMetricCore } from "./base-metrics";

// Canonical helper lives in test-case (used by serialization boundaries too).
export { resolveRetrievalContext } from "../test-case";

/**
 * Resolve a metric's `model` option into a concrete model.
 *
 * TS counterpart of Python's `initialize_model`. Every DeepEval TS model returns
 * `{ output, cost }`, so they are all "native" (cost is always accrued).
 */
export function initializeModel(model?: DeepEvalBaseLLM | string): {
  model: DeepEvalBaseLLM;
  usingNativeModel: boolean;
} {
  if (model instanceof DeepEvalBaseLLM) {
    return { model, usingNativeModel: true };
  }
  if (typeof model === "string") {
    return { model: new OpenAIModel({ model }), usingNativeModel: true };
  }
  return { model: new OpenAIModel(), usingNativeModel: true };
}

/**
 * Render an LLM call against a zod schema and accrue its cost onto the metric.
 * Returns the validated, typed object (the TS analogue of Python's
 * `generate_with_schema_and_extract`).
 */
export async function generateWithSchema<T>(
  metric: BaseMetricCore,
  prompt: string,
  schema: ZodType<T>,
): Promise<T> {
  if (!metric.model) {
    throw new Error("This metric has no model configured.");
  }
  const { output, cost } = await metric.model.generate(prompt, schema);
  metric.accrueCost(cost);
  return output;
}

// --- Required-param validation (centralized, enum-driven) ------------------

/** Maps an `SingleTurnParams` enum value to its accessor on `LLMTestCase`. */
const LLM_TEST_CASE_PARAM_GETTERS: Record<
  SingleTurnParams,
  (testCase: LLMTestCase) => unknown
> = {
  [SingleTurnParams.INPUT]: (tc) => tc.input,
  [SingleTurnParams.ACTUAL_OUTPUT]: (tc) => tc.actualOutput,
  [SingleTurnParams.EXPECTED_OUTPUT]: (tc) => tc.expectedOutput,
  [SingleTurnParams.CONTEXT]: (tc) => tc.context,
  [SingleTurnParams.RETRIEVAL_CONTEXT]: (tc) => tc.retrievalContext,
  [SingleTurnParams.TOOLS_CALLED]: (tc) => tc.toolsCalled,
  [SingleTurnParams.EXPECTED_TOOLS]: (tc) => tc.expectedTools,
  [SingleTurnParams.MCP_SERVERS]: (tc) => tc.mcpServers,
  [SingleTurnParams.MCP_TOOLS_CALLED]: (tc) => tc.mcpToolsCalled,
  [SingleTurnParams.MCP_RESOURCES_CALLED]: (tc) => tc.mcpResourcesCalled,
  [SingleTurnParams.MCP_PROMPTS_CALLED]: (tc) => tc.mcpPromptsCalled,
};

function joinMissingParams(params: string[]): string {
  if (params.length === 1) return params[0];
  if (params.length === 2) return params.join(" and ");
  return `${params.slice(0, -1).join(", ")}, and ${params[params.length - 1]}`;
}

/**
 * Verify a test case provides every param a metric requires. Centralizes what
 * Python's `check_llm_test_case_params` does: drives off the metric's
 * `requiredParams` enum list, sets `metric.error`, and throws
 * `MissingTestCaseParamsError` (which the evaluate() runner can skip on).
 */
export function checkSingleTurnParams(
  testCase: LLMTestCase,
  requiredParams: SingleTurnParams[],
  metric: BaseMetricCore,
): void {
  if (!(testCase instanceof LLMTestCase)) {
    const err = `Unable to evaluate test cases that are not of type 'LLMTestCase' using the '${metric.name}' metric.`;
    metric.error = err;
    throw new DeepEvalError(err);
  }

  if (
    requiredParams.includes(SingleTurnParams.ACTUAL_OUTPUT) &&
    testCase.actualOutput === ""
  ) {
    const err = `'actual_output' cannot be empty for the '${metric.name}' metric`;
    metric.error = err;
    throw new MissingTestCaseParamsError(err);
  }

  const missing = requiredParams
    .filter((p) => LLM_TEST_CASE_PARAM_GETTERS[p](testCase) == null)
    .map((p) => `'${p}'`);

  if (missing.length > 0) {
    const err = `${joinMissingParams(missing)} cannot be None for the '${metric.name}' metric`;
    metric.error = err;
    throw new MissingTestCaseParamsError(err);
  }
}

/**
 * Validate an `ArenaTestCase`: all contestants share the same input/expected
 * output, and each contestant's test case provides the required params.
 * Mirrors Python's `check_arena_test_case_params`.
 */
export function checkArenaTestCaseParams(
  arenaTestCase: ArenaTestCase,
  requiredParams: SingleTurnParams[],
  metric: BaseMetricCore,
): void {
  const cases = arenaTestCase.contestants.map((c) => c.testCase);
  const refInput = cases[0].input;
  if (cases.slice(1).some((c) => c.input !== refInput)) {
    throw new TypeError("All contestants must have the same 'input'.");
  }
  const refExpected = cases[0].expectedOutput;
  if (cases.slice(1).some((c) => c.expectedOutput !== refExpected)) {
    throw new TypeError("All contestants must have the same 'expectedOutput'.");
  }
  for (const tc of cases) {
    checkSingleTurnParams(tc, requiredParams, metric);
  }
}

/** Format a list of tool calls as an indented JSON array (for tool-metric prompts). */
export function printToolsCalled(tools: ToolCall[]): string {
  if (!tools || tools.length === 0) return "";
  const parts = tools.map((t) => {
    const json = JSON.stringify(
      {
        name: t.name,
        description: t.description,
        reasoning: t.reasoning,
        output: t.output,
        inputParameters: t.inputParameters,
      },
      null,
      4,
    );
    return json
      .split("\n")
      .map((line) => "  " + line)
      .join("\n");
  });
  return "[\n" + parts.join(",\n") + "\n]";
}

// --- Verbose logs (centralized formatting) ---------------------------------

/** Pretty-print a list for verbose logs (strings quoted, objects JSON-indented). */
export function prettifyList(items: unknown[]): string {
  if (items.length === 0) return "[]";
  const formatted = items.map((item) =>
    typeof item === "string"
      ? `"${item}"`
      : JSON.stringify(item, null, 4).replace(/\n/g, "\n    "),
  );
  return `[\n    ${formatted.join(",\n    ")}\n]`;
}

/**
 * Build (and, when `verboseMode`, print) a metric's verbose logs from its steps.
 * Mirrors Python's `construct_verbose_logs`: stores all-but-last step, prints the
 * full set. Returns the stored string for `metric.verboseLogs`.
 */
export function constructVerboseLogs(
  metric: BaseMetricCore,
  steps: string[],
): string {
  let logs = "";
  for (let i = 0; i < steps.length - 1; i++) {
    logs += steps[i];
    if (i < steps.length - 2) logs += " \n \n";
  }
  if (metric.verboseMode && steps.length > 0) {
    const full = `${logs} \n \n${steps[steps.length - 1]}`;
    console.log(
      `\n${"=".repeat(70)}\n${metric.name} Verbose Logs\n${"=".repeat(70)}\n${full}\n`,
    );
  }
  return logs;
}
