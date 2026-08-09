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
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "@/metrics/utils";
import { reprPrimitive, indentMultilineString } from "@/metrics/mcp/utils";
import {
  MCPPrimitivesScoreSchema,
  MCPArgsScoreSchema,
} from "@/metrics/mcp-use-metric/schema";
import { type MetricTemplateOverride } from "@/templates/override";

const TEMPLATE_CLASS = "MCPUseMetric";

export type MCPUseTemplateOverride = MetricTemplateOverride<"MCPUseMetric">;

export interface MCPUseMetricOptions {
  threshold?: number | null;
  flaky?: boolean;
  model?: DeepEvalBaseLLM | string;
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
    const { model, usingNativeModel } = initializeModel(options.model);
    this.model = model;
    this.usingNativeModel = usingNativeModel;
    this.evaluationModel = this.model.getModelName();
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    this.error = undefined;
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = this.usingNativeModel ? 0 : undefined;

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

      const primScore = await generateWithSchema(
        this,
        this.getPrompt("get_primitive_correctness_prompt", {
          test_case: testCaseVars,
          available_primitives: availablePrimitives,
          primitives_used: primitivesUsed,
        }),
        MCPPrimitivesScoreSchema,
      );
      const argScore = await generateWithSchema(
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

  get name(): string {
    return "MCP Use";
  }
}
