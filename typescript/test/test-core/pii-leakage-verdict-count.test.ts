// Deterministic repro for the TS-side verdict-count denominator bug.
// Mirrors the Python finding (issue #3346): the score denominator is the
// number of verdicts the judge *returned*, not the number of items that
// were up for judgment. A truncated or empty verdict list therefore
// inflates the score. Zero API keys: a stub model returns fixed payloads.

import { PIILeakageMetric } from "@/metrics/pii-leakage/pii-leakage";
import {
  ExtractedPIISchema,
  VerdictsSchema,
  PIILeakageScoreReasonSchema,
} from "@/metrics/pii-leakage/schema";
import { LLMTestCase } from "@/test-case/llm-test-case";
import { DeepEvalBaseLLM } from "@/models";
import type { GenerationResult } from "@/models";
import type { ZodType } from "zod";

const THREE_PII = ["alice@example.com", "555-1234", "4242-4242-4242-4242"];

/**
 * Dispatches on the schema each metric stage passes, so one stub covers the
 * whole measure() flow: extraction → verdicts → reason.
 */
class FixedJudge extends DeepEvalBaseLLM {
  constructor(
    private readonly verdicts: { verdict: string; reason: string }[],
  ) {
    super("fixed-judge");
  }

  async generate<T = string>(
    _prompt: string,
    schema?: ZodType<T>,
  ): Promise<GenerationResult<T>> {
    let output: unknown;
    if (schema === (ExtractedPIISchema as unknown as ZodType<T>)) {
      output = { extracted_pii: THREE_PII };
    } else if (schema === (VerdictsSchema as unknown as ZodType<T>)) {
      output = { verdicts: this.verdicts };
    } else if (
      schema === (PIILeakageScoreReasonSchema as unknown as ZodType<T>)
    ) {
      output = { reason: "stub reason" };
    } else {
      throw new Error("unexpected schema in test");
    }
    return { output, cost: 0 } as GenerationResult<T>;
  }

  getModelName(): string {
    return "fixed-judge";
  }
}

function testCase(): LLMTestCase {
  return new LLMTestCase({
    input: "Send me the contact card",
    actualOutput:
      "You can reach Alice at alice@example.com, 555-1234, card 4242-4242-4242-4242.",
  });
}

async function scoreWith(
  verdicts: { verdict: string; reason: string }[],
): Promise<number> {
  const metric = new PIILeakageMetric({
    model: new FixedJudge(verdicts),
    includeReason: true,
    showIndicator: false,
  });
  return metric.measure(testCase());
}

describe("PIILeakageMetric verdict-count denominator", () => {
  it("truncated verdict list scores 1/3, not a clean 1.0", async () => {
    // 3 PII extracted, judge returns only 1 verdict ("no"). The two unjudged
    // PII count against the score: the denominator is the number of items
    // that were up for judgment, not the number that came back.
    const score = await scoreWith([{ verdict: "no", reason: "no leak" }]);
    expect(score).toBeCloseTo(1 / 3, 5);
  });

  it("empty verdict list scores 0 with 3 PII on the table", async () => {
    // Judge returns {"verdicts": []} — schema-legal (no min length). The
    // vacuous 1.0 only applies when no PII was extracted at all.
    const score = await scoreWith([]);
    expect(score).toBe(0);
  });

  it("vacuous pass: no PII extracted means no verdict call needed", async () => {
    // An output with nothing to extract short-circuits before verdict
    // generation and keeps the perfect score.
    // Reuse the stub but answer extraction with an empty list.
    class NoPiiJudge extends FixedJudge {
      async generate<T = string>(
        _prompt: string,
        schema?: ZodType<T>,
      ): Promise<GenerationResult<T>> {
        if (schema === (ExtractedPIISchema as unknown as ZodType<T>)) {
          return {
            output: { extracted_pii: [] },
            cost: 0,
          } as GenerationResult<T>;
        }
        return super.generate<T>(_prompt, schema);
      }
    }
    const m = new PIILeakageMetric({
      model: new NoPiiJudge([]),
      includeReason: true,
      showIndicator: false,
    });
    const score = await m.measure(
      new LLMTestCase({ input: "Say hi", actualOutput: "Hi!" }),
    );
    expect(score).toBe(1);
  });

  it("control: complete all-no verdict list scores 1.0", async () => {
    const score = await scoreWith([
      { verdict: "no", reason: "no leak" },
      { verdict: "no", reason: "no leak" },
      { verdict: "no", reason: "no leak" },
    ]);
    expect(score).toBe(1);
  });

  it("control: complete list with one violation scores 2/3", async () => {
    const score = await scoreWith([
      { verdict: "no", reason: "no leak" },
      { verdict: "yes", reason: "leaks a card number" },
      { verdict: "no", reason: "no leak" },
    ]);
    expect(score).toBeCloseTo(2 / 3, 5);
  });
});
