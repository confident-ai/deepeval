import type { ZodType } from "zod";
import {
  AmazonBedrockModel,
  AnthropicModel,
  AzureOpenAIModel,
  DeepEvalBaseLLM,
  DeepSeekModel,
  GeminiModel,
  GrokModel,
  KimiModel,
  LocalModel,
  OllamaModel,
  OpenAIModel,
  OpenRouterModel,
  PortkeyModel,
} from "@/models";
import { selectProvider, type ProviderId } from "@/models/provider-selection";
import {
  LLMTestCase,
  SingleTurnParams,
  ToolCall,
  ArenaTestCase,
} from "@/test-case";
import { DeepEvalError, MissingTestCaseParamsError } from "@/errors";
import { extractJson } from "@/models/utils";
import { BaseMetricCore } from "@/metrics/base-metrics";
import {
  EVAL_MODE_ENV_VAR,
  EvalMode,
  resolveEvalMode,
  usesSystemOne,
  type EvalModeName,
} from "@/config/eval-mode";
import { DeepEvalBaseSystemOneModel } from "@/models/system-one/base-system-one-model";
import { TypeSafeModel } from "@/models/system-one/typesafe-model";

// Canonical helper lives in test-case (used by serialization boundaries too).
export { resolveRetrievalContext } from "@/test-case";

const MODEL_BY_PROVIDER: Record<
  ProviderId,
  (model?: string) => DeepEvalBaseLLM
> = {
  openai: (model) => new OpenAIModel({ model }),
  gemini: (model) => new GeminiModel({ model }),
  portkey: (model) => new PortkeyModel({ model }),
  ollama: (model) => new OllamaModel({ model }),
  "local-model": (model) => new LocalModel({ model }),
  "azure-openai": (model) => new AzureOpenAIModel({ model }),
  moonshot: (model) => new KimiModel({ model }),
  grok: (model) => new GrokModel({ model }),
  deepseek: (model) => new DeepSeekModel({ model }),
  openrouter: (model) => new OpenRouterModel({ model }),
  anthropic: (model) => new AnthropicModel({ model }),
  bedrock: (model) => new AmazonBedrockModel({ model }),
};

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

  const modelName = typeof model === "string" ? model : undefined;
  const provider = selectProvider();
  const build = provider
    ? MODEL_BY_PROVIDER[provider]
    : MODEL_BY_PROVIDER.openai;

  return { model: build(modelName), usingNativeModel: true };
}

/**
 * Build the System One model a metric decides with, or `undefined` when its
 * eval mode never calls one. Under `llm` nothing is built, so a missing
 * TypeSafe key is not an error; under `hybrid` and `system_one` a missing key
 * fails here, at construction, rather than mid-evaluation.
 */
export function initializeSystemOneModel(
  model: DeepEvalBaseSystemOneModel | string | undefined,
  evalMode: EvalModeName,
): DeepEvalBaseSystemOneModel | undefined {
  if (!usesSystemOne(evalMode)) return undefined;
  if (model instanceof DeepEvalBaseSystemOneModel) return model;
  if (model !== undefined && typeof model !== "string") {
    throw new TypeError(
      "Unsupported type for systemOneModel. Expected undefined, a string, " +
        "or a DeepEvalBaseSystemOneModel.",
    );
  }
  try {
    return new TypeSafeModel({ model });
  } catch (e) {
    throw new DeepEvalError(
      `${EVAL_MODE_ENV_VAR}=${evalMode} routes metric decisions to TypeSafe ` +
        `AI Jev, but it is not usable: ${(e as Error).message} Configure it ` +
        "with `npx deepeval set-typesafe --prompt-api-key` or switch back " +
        `with \`npx deepeval set-eval-mode ${EvalMode.LLM}\`.`,
    );
  }
}

export interface MetricModelOptions {
  model?: DeepEvalBaseLLM | string;
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  evalMode?: EvalModeName;
  /**
   * `false` for a metric that never asks Jev (JSON Correctness): no System
   * One model is built in any mode, and `system_one` only means "no LLM".
   */
  systemOne?: boolean;
}

/**
 * Resolve a metric's eval mode and both of its models in one place. Under
 * `system_one` a metric with a whole-metric form builds no LLM (Jev runs the
 * whole metric and nothing falls back), so a missing LLM key is not an error.
 * A metric without one (DAG, whose task nodes need the LLM) runs as `hybrid`
 * there and always gets its LLM. Metrics that take no `systemOneModel` option
 * get the default TypeSafe model when their mode needs one.
 */
export function initializeMetricModels(
  metric: BaseMetricCore,
  options: MetricModelOptions,
): void {
  const evalMode = resolveEvalMode(options.evalMode);
  const asksJev = options.systemOne ?? true;
  metric.evalMode = evalMode;
  metric.systemOneModel = asksJev
    ? initializeSystemOneModel(options.systemOneModel, evalMode)
    : undefined;
  const hasWholeMetricForm =
    metric.systemOneEvalSpec !== BaseMetricCore.prototype.systemOneEvalSpec;
  if (evalMode === EvalMode.SYSTEM_ONE && (hasWholeMetricForm || !asksJev)) {
    metric.model = undefined;
    metric.usingNativeModel = true;
    metric.evaluationModel = metric.systemOneModel?.getModelName();
    return;
  }
  const { model, usingNativeModel } = initializeModel(options.model);
  metric.model = model;
  metric.usingNativeModel = usingNativeModel;
  metric.evaluationModel = model.getModelName();
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
    if (metric.evalMode === EvalMode.SYSTEM_ONE) {
      throw new DeepEvalError(
        `${metric.name} runs on System One under ${EVAL_MODE_ENV_VAR}=` +
          `${EvalMode.SYSTEM_ONE} and has no LLM for this step. Switch it ` +
          `back with \`evalMode: "${EvalMode.LLM}"\` or ` +
          `\`npx deepeval set-eval-mode ${EvalMode.LLM}\`.`,
      );
    }
    throw new Error("This metric has no model configured.");
  }
  const { output, cost } = await metric.model.generate(prompt, schema);
  metric.accrueCost(cost);

  // Every shipped provider parses against the schema itself, so this is a
  // no-op for them. A custom model that ignores the `schema` argument hands
  // back raw text instead, which every metric would otherwise destructure into
  // `undefined` and fail on far from the cause.
  const parsed = schema.safeParse(output);
  if (parsed.success) return parsed.data;

  if (typeof output === "string") {
    try {
      return schema.parse(extractJson(output));
    } catch {
      // Fall through to the error below, which names the real problem.
    }
  }

  throw new DeepEvalError(
    `The evaluation model '${metric.model.getModelName()}' did not return output matching the schema ` +
      `the '${metric.name}' metric asked for. A custom model's \`generate\` must honor its \`schema\` ` +
      `argument and return the parsed object as \`output\`.`,
  );
}

/** A run needs one non-flaky metric with a threshold, else nothing votes. */
export function checkAtLeastOneMetricHasThreshold(
  metrics: BaseMetricCore[],
): void {
  if (metrics.some((m) => m.threshold !== null && !m.flaky)) return;
  throw new DeepEvalError(
    "You must provide at least one non-flaky metric with a 'threshold', " +
      "otherwise test cases can never pass or fail.",
  );
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

/**
 * Refuse a multimodal test case on a model that cannot see images, as Python's
 * `check_llm_test_case_params` does. An unknown capability counts as "cannot":
 * the alternative is a request whose images are silently dropped, scoring the
 * judge on image slugs it read as literal text.
 */
export function checkMultimodalSupport(metric: BaseMetricCore): void {
  if (!metric.multimodal) return;
  const model = metric.model;
  if (model?.supportsMultimodal()) return;

  if (!model) {
    // A metric has no LLM only when a System One model (Jev) is judging it,
    // and Jev reads text only.
    const err = metric.systemOneModel
      ? `${metric.name} is judged by System One (Jev) in this eval mode, and ` +
        `Jev evaluates text only. Run multimodal test cases with ` +
        `\`evalMode: "llm"\` or \`evalMode: "hybrid"\`.`
      : `The '${metric.name}' metric has no evaluation model and cannot evaluate multimodal test cases.`;
    metric.error = err;
    throw new DeepEvalError(err);
  }

  // Listed in full, as Python does — any subset would be an arbitrary
  // recommendation, and registry order puts the oldest models first.
  const alternatives = model.multimodalAlternatives();
  const suggestion =
    alternatives.length > 0
      ? ` Vision-capable models for this provider include ${alternatives.join(", ")}.`
      : "";
  const err =
    `The evaluation model '${model.getModelName()}' does not support multimodal evaluations, ` +
    `which the '${metric.name}' metric needs for this test case.${suggestion}`;
  metric.error = err;
  throw new DeepEvalError(err);
}

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

  metric.multimodal = testCase.multimodal;
  checkMultimodalSupport(metric);

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

// Metrics whose score direction was inverted; each warns once per process so a
// run that builds the metric per test case doesn't repeat itself.
const scoreDirectionWarned = new Set<string>();

export function warnScoreDirectionFlipped(metricName: string): void {
  if (scoreDirectionWarned.has(metricName)) return;
  scoreDirectionWarned.add(metricName);
  console.warn(
    `[deepeval] '${metricName}' now scores in the same direction as every other ` +
      "deepeval metric: 1 is a pass, 0 is a failure, and 'threshold' is the " +
      "MINIMUM passing score. It previously scored the proportion of " +
      "violations, where 'threshold' was a maximum. Review any 'threshold' you " +
      "pass and any code reading '.score' — a threshold of 0.2 that used to " +
      "mean 'at most 20% violations' should now be 0.8. This notice will be " +
      "removed in a future release.",
  );
}
