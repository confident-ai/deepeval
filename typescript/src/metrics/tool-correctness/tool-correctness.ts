import { BaseMetric } from "../base-metrics";
import {
  LLMTestCase,
  SingleTurnParams,
  ToolCallParams,
  ToolCall,
} from "../../test-case";
import { DeepEvalBaseLLM } from "../../models";
import { resolveTemplate } from "../../templates";
import {
  initializeModel,
  generateWithSchema,
  checkSingleTurnParams,
  constructVerboseLogs,
  printToolsCalled,
} from "../utils";
import { ToolSelectionScoreSchema, type ToolSelectionScore } from "./schema";

const TEMPLATE_CLASS = "ToolCorrectnessMetric";

/** Order-insensitive deep equality (matches Python `==` on dicts/values). */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null) return a === b;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]));
  }
  if (typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a as object);
    const kb = Object.keys(b as object);
    if (ka.length !== kb.length) return false;
    return ka.every((k) =>
      deepEqual(
        (a as Record<string, unknown>)[k],
        (b as Record<string, unknown>)[k],
      ),
    );
  }
  return false;
}

/** ToolCall equality: same name, input parameters, and output (Python `__eq__`). */
function toolCallEquals(a: ToolCall, b: ToolCall): boolean {
  return (
    a.name === b.name &&
    deepEqual(a.inputParameters, b.inputParameters) &&
    deepEqual(a.output, b.output)
  );
}

/** Dedup a list of names, preserving Python `set()`-style membership. */
function uniqueMissing(expected: string[], called: string[]): string[] {
  const calledSet = new Set(called);
  return [...new Set(expected)].filter((n) => !calledSet.has(n));
}

export interface ToolCorrectnessMetricOptions {
  /** If provided (and non-empty), an LLM also judges tool *selection*. */
  availableTools?: ToolCall[];
  threshold?: number;
  /** Which `ToolCall` fields to compare (input parameters / output). */
  evaluationParams?: ToolCallParams[];
  model?: DeepEvalBaseLLM | string;
  includeReason?: boolean;
  strictMode?: boolean;
  verboseMode?: boolean;
  showIndicator?: boolean;
  shouldExactMatch?: boolean;
  shouldConsiderOrdering?: boolean;
}

/**
 * Tool Correctness — were the right tools called (vs. `expectedTools`)?
 * Deterministic tool-calling score (exact / ordered / unordered), combined via
 * `min` with an optional LLM tool-*selection* score when `availableTools` is set.
 */
export class ToolCorrectnessMetric extends BaseMetric {
  toolsCalled: ToolCall[] = [];
  expectedTools: ToolCall[] = [];
  private readonly availableTools?: ToolCall[];
  private readonly evaluationParams: ToolCallParams[];
  private readonly shouldExactMatch: boolean;
  private readonly shouldConsiderOrdering: boolean;

  constructor(options: ToolCorrectnessMetricOptions = {}) {
    const strictMode = options.strictMode ?? false;
    super(strictMode ? 1 : (options.threshold ?? 0.5), {
      strictMode,
      verboseMode: options.verboseMode,
      includeReason: options.includeReason ?? true,
      showIndicator: options.showIndicator,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.TOOLS_CALLED,
      SingleTurnParams.EXPECTED_TOOLS,
    ];
    this.availableTools = options.availableTools;
    this.evaluationParams = options.evaluationParams ?? [];
    this.shouldExactMatch = options.shouldExactMatch ?? false;
    this.shouldConsiderOrdering = options.shouldConsiderOrdering ?? false;
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

      this.toolsCalled = testCase.toolsCalled ?? [];
      this.expectedTools = testCase.expectedTools ?? [];

      const toolCallingScore = this.calculateScore();
      const toolSelectionScore: ToolSelectionScore =
        this.availableTools && this.availableTools.length > 0
          ? await this.getToolSelectionScore(testCase.input)
          : {
              score: 1,
              reason:
                "No available tools were provided to assess tool selection criteria",
            };

      const combined = Math.min(toolCallingScore, toolSelectionScore.score);
      this.score =
        this.strictMode && combined < this.threshold ? 0 : combined;
      this.reason = this.constructFinalReason(
        this.generateReason(),
        toolSelectionScore.reason,
      );
      this.success = this.score >= this.threshold;

      this.verboseLogs = constructVerboseLogs(this, [
        `Expected Tools:\n${printToolsCalled(this.expectedTools)}`,
        `Tools Called:\n${printToolsCalled(this.toolsCalled)}`,
        `Available Tools:\n${this.availableTools ? printToolsCalled(this.availableTools) : "[]"}`,
        `Tool Selection Score: ${toolSelectionScore.score}`,
        `Tool Selection Reason: ${toolSelectionScore.reason}`,
        `Final Score: ${this.score}\nFinal Reason: ${this.reason}`,
      ]);
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  private async getToolSelectionScore(
    userInput: string,
  ): Promise<ToolSelectionScore> {
    const prompt = resolveTemplate("metrics", TEMPLATE_CLASS, "get_tool_selection_score", {
      user_input: userInput,
      tools_called: printToolsCalled(this.toolsCalled),
      available_tools: printToolsCalled(this.availableTools ?? []),
    });
    return generateWithSchema(this, prompt, ToolSelectionScoreSchema);
  }

  // --- scoring ---

  private calculateScore(): number {
    let score: number;
    if (this.shouldExactMatch) {
      score = this.calculateExactMatchScore();
    } else if (this.shouldConsiderOrdering) {
      const { weightedLength } = this.computeWeightedLcs();
      if (this.toolsCalled.length === 0 && this.expectedTools.length === 0) {
        score = 1;
      } else if (this.expectedTools.length === 0) {
        score = 0;
      } else {
        score = weightedLength / this.expectedTools.length;
      }
    } else {
      score = this.calculateNonExactMatchScore();
    }
    return this.strictMode && score < this.threshold ? 0 : score;
  }

  private calculateExactMatchScore(): number {
    if (this.toolsCalled.length !== this.expectedTools.length) return 0;
    if (this.expectedTools.length === 0) return 1;
    for (let i = 0; i < this.toolsCalled.length; i++) {
      const called = this.toolsCalled[i];
      const expected = this.expectedTools[i];
      if (called.name !== expected.name) return 0;
      if (
        this.evaluationParams.includes(ToolCallParams.INPUT_PARAMETERS) &&
        !deepEqual(called.inputParameters, expected.inputParameters)
      ) {
        return 0;
      }
      if (
        this.evaluationParams.includes(ToolCallParams.OUTPUT) &&
        !deepEqual(called.output, expected.output)
      ) {
        return 0;
      }
    }
    return 1;
  }

  private calculateNonExactMatchScore(): number {
    let totalScore = 0;
    const matchedCalled = new Set<number>();
    for (const expected of this.expectedTools) {
      let bestScore = 0;
      let bestIdx = -1;
      for (let j = 0; j < this.toolsCalled.length; j++) {
        if (matchedCalled.has(j)) continue;
        const called = this.toolsCalled[j];
        if (expected.name !== called.name) continue;
        let matchScore = 1;
        if (this.evaluationParams.includes(ToolCallParams.INPUT_PARAMETERS)) {
          matchScore *= this.compareDicts(
            expected.inputParameters ?? {},
            called.inputParameters ?? {},
          );
        }
        if (
          this.evaluationParams.includes(ToolCallParams.OUTPUT) &&
          !deepEqual(expected.output, called.output)
        ) {
          matchScore = 0;
        }
        if (matchScore > bestScore) {
          bestScore = matchScore;
          bestIdx = j;
        }
      }
      if (bestScore > 0) {
        totalScore += bestScore;
        matchedCalled.add(bestIdx);
      }
    }
    if (this.expectedTools.length === 0 && this.toolsCalled.length === 0)
      return 1;
    if (this.expectedTools.length === 0) return 0;
    return totalScore / this.expectedTools.length;
  }

  private computeWeightedLcs(): { lcs: ToolCall[]; weightedLength: number } {
    const expected = this.expectedTools;
    const called = this.toolsCalled;
    const m = expected.length;
    const n = called.length;
    const dp: number[][] = Array.from({ length: m + 1 }, () =>
      new Array(n + 1).fill(0),
    );
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const e = expected[i - 1];
        const c = called[j - 1];
        if (e.name !== c.name) {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
          continue;
        }
        let score = 1;
        if (this.evaluationParams.includes(ToolCallParams.INPUT_PARAMETERS)) {
          score *= this.compareDicts(
            e.inputParameters ?? {},
            c.inputParameters ?? {},
          );
        }
        if (
          this.evaluationParams.includes(ToolCallParams.OUTPUT) &&
          !deepEqual(e.output, c.output)
        ) {
          score = 0;
        }
        dp[i][j] = Math.max(
          dp[i - 1][j],
          dp[i][j - 1],
          score > 0 ? dp[i - 1][j - 1] + score : 0,
        );
      }
    }
    let i = m;
    let j = n;
    const lcs: ToolCall[] = [];
    while (i > 0 && j > 0) {
      if (dp[i][j] === dp[i - 1][j]) {
        i -= 1;
      } else if (dp[i][j] === dp[i][j - 1]) {
        j -= 1;
      } else {
        lcs.push(expected[i - 1]);
        i -= 1;
        j -= 1;
      }
    }
    lcs.reverse();
    return { lcs, weightedLength: dp[m][n] };
  }

  private compareDicts(
    d1: Record<string, unknown>,
    d2: Record<string, unknown>,
  ): number {
    if (deepEqual(d1, d2)) return 1;
    if (this.shouldExactMatch) return 0;
    const keys1 = Object.keys(d1);
    const keys2 = Object.keys(d2);
    const matchedKeys = keys1.filter((k) => keys2.includes(k));
    const totalKeys = new Set([...keys1, ...keys2]).size;
    if (totalKeys === 0) return 1;
    let matchScore = 0;
    for (const key of matchedKeys) {
      if (deepEqual(d1[key], d2[key])) {
        matchScore += 1 / totalKeys;
      } else if (
        d1[key] != null &&
        d2[key] != null &&
        typeof d1[key] === "object" &&
        typeof d2[key] === "object" &&
        !Array.isArray(d1[key]) &&
        !Array.isArray(d2[key])
      ) {
        matchScore +=
          this.compareDicts(
            d1[key] as Record<string, unknown>,
            d2[key] as Record<string, unknown>,
          ) / totalKeys;
      }
    }
    return matchScore;
  }

  // --- deterministic tool-calling reason ---

  private generateReason(): string {
    const calledNames = this.toolsCalled.map((t) => t.name);
    const expectedNames = this.expectedTools.map((t) => t.name);

    if (this.shouldExactMatch) {
      const label = this.calculateExactMatchScore()
        ? "Exact match"
        : "Not an exact match";
      return `${label}: expected ${JSON.stringify(expectedNames)}, called ${JSON.stringify(calledNames)}. See details above.`;
    }

    if (this.shouldConsiderOrdering) {
      const { lcs, weightedLength } = this.computeWeightedLcs();
      let score: number;
      if (this.toolsCalled.length === 0 && this.expectedTools.length === 0)
        score = 1;
      else if (this.expectedTools.length === 0) score = 0;
      else score = weightedLength / this.expectedTools.length;

      if (score === 1) {
        return `Correct ordering: all expected tools ${JSON.stringify(expectedNames)} were called in the correct order.`;
      }
      const missing = uniqueMissing(expectedNames, calledNames);
      const outOfOrder = uniqueMissing(
        expectedNames,
        lcs.map((t) => t.name),
      );
      const issues: string[] = [];
      if (missing.length) issues.push(`missing tools ${JSON.stringify(missing)}`);
      if (outOfOrder.length)
        issues.push(`out-of-order tools ${JSON.stringify(outOfOrder)}`);
      return `Incorrect tool usage: ${issues.join(" and ")}; expected ${JSON.stringify(expectedNames)}, called ${JSON.stringify(calledNames)}. See more details above.`;
    }

    if (this.calculateNonExactMatchScore() === 1) {
      return `All expected tools ${JSON.stringify(expectedNames)} were called (order not considered).`;
    }
    const missing = this.expectedTools
      .filter((e) => !this.toolsCalled.some((c) => toolCallEquals(c, e)))
      .map((t) => t.name);
    return `Incomplete tool usage: missing tools ${JSON.stringify(missing)}; expected ${JSON.stringify(expectedNames)}, called ${JSON.stringify(calledNames)}. See more details above.`;
  }

  private constructFinalReason(
    toolCallingReason: string,
    toolSelectionReason: string,
  ): string {
    return (
      "[\n" +
      "\t Tool Calling Reason: " +
      toolCallingReason +
      "\n" +
      "\t Tool Selection Reason: " +
      toolSelectionReason +
      "\n" +
      "]\n"
    );
  }

  isSuccessful(): boolean {
    const ok = this.error == null && (this.score ?? 0) >= this.threshold;
    this.success = ok;
    return ok;
  }

  get name(): string {
    return "Tool Correctness";
  }
}
