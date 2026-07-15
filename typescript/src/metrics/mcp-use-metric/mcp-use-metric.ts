import { BaseMetric } from "../base-metrics";
import {
  LLMTestCase,
  SingleTurnParams,
  MCPServer,
  MCPToolCall,
  MCPResourceCall,
  MCPPromptCall,
} from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
} from "../utils";
import { reprPrimitive, indentMultilineString } from "../mcp/utils";
import { MCPPrimitivesScoreSchema, MCPArgsScoreSchema } from "./schema";

const TEMPLATE_CLASS = "MCPUseMetric";

export interface MCPUseMetricOptions {
  threshold?: number;
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
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
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
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
          testCase.mcpToolsCalled ?? [],
          testCase.mcpResourcesCalled ?? [],
          testCase.mcpPromptsCalled ?? [],
        );
      const testCaseVars = {
        input: testCase.input,
        actual_output: testCase.actualOutput,
      };

      const primScore = await generateWithSchema(
        this,
        resolveTemplate("metrics", TEMPLATE_CLASS, "get_primitive_correctness_prompt", {
          test_case: testCaseVars,
          available_primitives: availablePrimitives,
          primitives_used: primitivesUsed,
        }),
        MCPPrimitivesScoreSchema,
      );
      const argScore = await generateWithSchema(
        this,
        resolveTemplate("metrics", TEMPLATE_CLASS, "get_mcp_argument_correctness_prompt", {
          test_case: testCaseVars,
          available_primitives: availablePrimitives,
          primitives_used: primitivesUsed,
        }),
        MCPArgsScoreSchema,
      );

      const score = Math.min(primScore.score, argScore.score);
      this.score = this.strictMode && score < this.threshold ? 0 : score;
      this.reason = this.includeReason
        ? `[\n\t${primScore.reason}\n\t${argScore.reason}\n]\n`
        : undefined;
      this.success = this.score >= this.threshold;

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
    mcpToolsCalled: MCPToolCall[],
    mcpResourcesCalled: MCPResourceCall[],
    mcpPromptsCalled: MCPPromptCall[],
  ): { availablePrimitives: string; primitivesUsed: string } {
    let availablePrimitives = "MCP Primitives Available: \n";
    for (const server of mcpServers) {
      availablePrimitives += `MCP Server ${server.serverName}\n`;
      availablePrimitives += block("Available Tools", server.availableTools ?? []);
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

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "MCP Use";
  }
}
