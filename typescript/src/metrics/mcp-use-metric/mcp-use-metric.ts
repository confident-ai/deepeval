import { BaseMetric, resolveThreshold } from "@/metrics/base-metrics";
import {
  LLMTestCase,
  SingleTurnParams,
  MCPServer,
  MCPToolCall,
  MCPResourceCall,
  MCPPromptCall,
  ToolCall,
} from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { DeepEvalBaseSystemOneModel } from "@/models/system-one";
import type { EvalModeName } from "@/config/eval-mode";
import {
  initializeMetricModels,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "@/metrics/utils";
import {
  reprPrimitive,
  indentMultilineString,
  mcpCallsState,
  mcpServersState,
} from "@/metrics/mcp/utils";
import {
  formatDecisionReason,
  parseQuestions,
  runSystemOneEval,
  systemOneScore,
  type SystemOneEvalSpec,
  type SystemOneScoreSpec,
} from "@/metrics/system-one";
import {
  MCPPrimitivesScoreSchema,
  MCPArgsScoreSchema,
} from "@/metrics/mcp-use-metric/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "MCPUseMetric";

export type MCPUseTemplateOverride = MetricTemplateOverride<"MCPUseMetric">;

const PRIMITIVE_USAGE_LEVELS = [
  "Wrong primitives",
  "Poor choice",
  "Reasonable choice",
  "Best choice",
];
const ARGUMENT_CORRECTNESS_LEVELS = [
  "Incorrect",
  "Mostly incorrect",
  "Mostly correct",
  "Fully correct",
];

export interface MCPUseMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
  /** The System One model (Jev) used under `hybrid` / `system_one`. */
  systemOneModel?: DeepEvalBaseSystemOneModel | string;
  /** Who decides; defaults to `DEEPEVAL_EVAL_MODE`, then `llm`. */
  evalMode?: EvalModeName;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  evaluationTemplate?: MCPUseTemplateOverride;
}

function block(label: string, items: unknown[]): string {
  if (!items || items.length === 0) return "";
  return (
    `\n${label}:\n[\n` +
    items.map((i) => indentMultilineString(reprPrimitive(i), 4)).join(",\n") +
    "\n]"
  );
}

/**
 * MCP Use — did the agent pick the right MCP primitives and pass correct
 * arguments? Scores primitive selection and argument correctness independently;
 * final score = min of the two. **Higher is better**. Requires `mcpServers`.
 */
export class MCPUseMetric extends BaseMetric {
  constructor(options: MCPUseMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : resolveThreshold(options.threshold, 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
      flaky: options.flaky,
      evaluationTemplate: options.evaluationTemplate,
    });
    this.multimodalAware = true;
    this.templateClass = TEMPLATE_CLASS;
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.MCP_SERVERS,
    ];
    initializeMetricModels(this, options);
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;
      if (await runSystemOneEval(this, testCase)) return this.score as number;

      const { availablePrimitives, primitivesUsed } =
        this.getMcpInteractionText(
          testCase.mcpServers ?? [],
          testCase.mcpToolsCalled ?? testCase.toolsCalled ?? [],
          testCase.mcpResourcesCalled ?? [],
          testCase.mcpPromptsCalled ?? [],
        );
      const testCaseVars = {
        input: testCase.input,
        actual_output: testCase.actualOutput,
      };

      const primValue = await systemOneScore(
        this,
        this.systemOnePrimitivesSpec(testCase),
      );
      const primScore =
        primValue !== undefined
          ? {
              score: primValue,
              reason: formatDecisionReason(this, "primitive usage", primValue),
            }
          : await generateWithSchema(
              this,
              this.getPrompt("get_primitive_correctness_prompt", {
                test_case: testCaseVars,
                available_primitives: availablePrimitives,
                primitives_used: primitivesUsed,
              }),
              MCPPrimitivesScoreSchema,
            );
      const argValue = await systemOneScore(
        this,
        this.systemOneArgsSpec(testCase),
      );
      const argScore =
        argValue !== undefined
          ? {
              score: argValue,
              reason: formatDecisionReason(
                this,
                "argument correctness",
                argValue,
              ),
            }
          : await generateWithSchema(
              this,
              this.getPrompt("get_mcp_argument_correctness_prompt", {
                test_case: testCaseVars,
                available_primitives: availablePrimitives,
                primitives_used: primitivesUsed,
              }),
              MCPArgsScoreSchema,
            );

      const score = Math.min(primScore.score, argScore.score);
      this.score = this.applyStrictMode(score);
      this.reason = this.includeReason
        ? `[\n\t${primScore.reason}\n\t${argScore.reason}\n]\n`
        : undefined;
      this.success = this.isSuccessful();

      this.verboseLogs = constructVerboseLogs(this, [
        availablePrimitives,
        primitivesUsed,
        `Primitive Usage Score: ${primScore.score}\nPrimitive Usage Reason: ${primScore.reason}`,
        `Argument Correctness Score: ${argScore.score}\nArgument Correctness Reason: ${argScore.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private getMcpInteractionText(
    mcpServers: MCPServer[],
    mcpToolsCalled: (MCPToolCall | ToolCall)[],
    mcpResourcesCalled: MCPResourceCall[],
    mcpPromptsCalled: MCPPromptCall[],
  ): { availablePrimitives: string; primitivesUsed: string } {
    let availablePrimitives = "MCP Primitives Available: \n";
    for (const server of mcpServers) {
      availablePrimitives += `MCP Server ${server.serverName}\n`;
      availablePrimitives += block(
        "Available Tools",
        server.availableTools ?? [],
      );
      availablePrimitives += block(
        "Available Resources",
        server.availableResources ?? [],
      );
      availablePrimitives += block(
        "Available Prompts",
        server.availablePrompts ?? [],
      );
    }
    let primitivesUsed = "MCP Primitives Used: \n";
    primitivesUsed += block("MCP Tools Called", mcpToolsCalled);
    primitivesUsed += block("MCP Resources Called", mcpResourcesCalled);
    primitivesUsed += block("MCP Prompts Called", mcpPromptsCalled);
    return { availablePrimitives, primitivesUsed };
  }

  private systemOneState(testCase: LLMTestCase): Record<string, unknown> {
    return {
      input: testCase.input,
      actual_output: testCase.actualOutput,
      mcp_servers: mcpServersState(testCase.mcpServers),
      primitives_used: mcpCallsState(
        testCase.mcpToolsCalled?.length
          ? testCase.mcpToolsCalled
          : (testCase.toolsCalled ?? []),
        testCase.mcpResourcesCalled ?? [],
        testCase.mcpPromptsCalled ?? [],
      ),
    };
  }

  private systemOnePrimitivesSpec(
    testCase: LLMTestCase,
  ): SystemOneScoreSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_primitive_score"),
      levels: PRIMITIVE_USAGE_LEVELS,
      state: this.systemOneState(testCase),
    };
  }

  private systemOneArgsSpec(
    testCase: LLMTestCase,
  ): SystemOneScoreSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      instructions: this.getPrompt("_experimental_system_one_args_score"),
      levels: ARGUMENT_CORRECTNESS_LEVELS,
      state: this.systemOneState(testCase),
    };
  }

  systemOneEvalSpec(testCase: LLMTestCase): SystemOneEvalSpec | undefined {
    if (testCase.multimodal) return undefined;
    return {
      evaluationParams: [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.MCP_TOOLS_CALLED,
        SingleTurnParams.MCP_RESOURCES_CALLED,
        SingleTurnParams.MCP_PROMPTS_CALLED,
        SingleTurnParams.TOOLS_CALLED,
      ],
      questions: parseQuestions(
        this.getPrompt("_experimental_system_one_questions"),
      ),
      extraState: {
        mcp_servers: mcpServersState(testCase.mcpServers),
      },
    };
  }

  get name(): string {
    return "MCP Use";
  }
}
