// Deterministic regression tests for TS verdict metrics OOV handling (issue #3391).
// Ensures PromptAlignmentMetric and RoleViolationMetric enforce valid verdict
// vocabularies ("yes" / "no") at schema parse time and scoring time. Zero API keys.

import { PromptAlignmentMetric } from "@/metrics/prompt-alignment/prompt-alignment";
import { RoleViolationMetric } from "@/metrics/role-violation/role-violation";
import { LLMTestCase } from "@/test-case/llm-test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { GenerationResult } from "@/models";
import type { ZodType } from "zod";

class StubJudge extends DeepEvalBaseLLM {
  constructor(private readonly payload: unknown) {
    super("stub-judge");
  }

  async generate<T = string>(
    _prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    let output = this.payload;
    if (schema) {
      const parsed = schema.safeParse(output);
      if (parsed.success) {
        output = parsed.data;
      }
    }
    return { output, cost: 0 } as GenerationResult<T>;
  }

  getModelName(): string {
    return "stub-judge";
  }
}

describe("TS verdict metrics OOV handling (#3391)", () => {
  describe("PromptAlignmentMetric", () => {
    it("rejects out-of-vocabulary verdicts at schema validation", async () => {
      const metric = new PromptAlignmentMetric({
        promptInstructions: ["Speak English."],
        model: new StubJudge({
          verdicts: [{ verdict: "n/a", reason: "ambiguous" }],
        }),
        includeReason: false,
        showIndicator: false,
      });

      const tc = new LLMTestCase({ input: "Hi", actualOutput: "Hello" });
      await expect(metric.measure(tc)).rejects.toThrow(
        /did not return output matching the schema/,
      );
    });

    it("positively scores 'yes' as aligned and 'no' as unaligned", async () => {
      const metric = new PromptAlignmentMetric({
        promptInstructions: ["Speak English.", "Be concise."],
        model: new StubJudge({
          verdicts: [
            { verdict: "yes", reason: null },
            { verdict: "no", reason: "Too verbose" },
          ],
        }),
        includeReason: false,
        showIndicator: false,
      });

      const tc = new LLMTestCase({ input: "Hi", actualOutput: "Hello" });
      const score = await metric.measure(tc);
      expect(score).toBeCloseTo(0.5, 5);
    });

    it("scores perfect 1.0 when all verdicts are 'yes'", async () => {
      const metric = new PromptAlignmentMetric({
        promptInstructions: ["Speak English."],
        model: new StubJudge({
          verdicts: [{ verdict: "yes", reason: null }],
        }),
        includeReason: false,
        showIndicator: false,
      });

      const tc = new LLMTestCase({ input: "Hi", actualOutput: "Hello" });
      const score = await metric.measure(tc);
      expect(score).toBe(1.0);
    });
  });

  describe("RoleViolationMetric", () => {
    it("rejects out-of-vocabulary verdicts at schema validation", async () => {
      const metric = new RoleViolationMetric({
        role: "helpful assistant",
        model: new StubJudge({
          role_violations: ["Pretended to be a lawyer"],
          verdicts: [{ verdict: "unknown", reason: "uncertain" }],
        }),
        includeReason: false,
        showIndicator: false,
      });

      const tc = new LLMTestCase({
        input: "Legal advice?",
        actualOutput: "Here is advice",
      });
      await expect(metric.measure(tc)).rejects.toThrow(
        /did not return output matching the schema/,
      );
    });

    it("scores clean pass (1.0) when all verdicts are 'no'", async () => {
      const metric = new RoleViolationMetric({
        role: "helpful assistant",
        model: new StubJudge({
          role_violations: ["Potential issue"],
          verdicts: [{ verdict: "no", reason: "Within boundaries" }],
        }),
        includeReason: false,
        showIndicator: false,
      });

      const tc = new LLMTestCase({ input: "Hi", actualOutput: "Hello" });
      const score = await metric.measure(tc);
      expect(score).toBe(1.0);
    });

    it("scores failed audit (0.0) when a violation verdict is 'yes'", async () => {
      const metric = new RoleViolationMetric({
        role: "helpful assistant",
        model: new StubJudge({
          role_violations: ["Broke character"],
          verdicts: [{ verdict: "yes", reason: "Claimed to be human" }],
        }),
        includeReason: false,
        showIndicator: false,
      });

      const tc = new LLMTestCase({
        input: "Who are you?",
        actualOutput: "I am a human",
      });
      const score = await metric.measure(tc);
      expect(score).toBe(0.0);
    });
  });
});
