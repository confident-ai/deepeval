// How metrics talk to models: schema tolerance, cost honesty, and GEval's
// log-prob-weighted scoring.

import { z } from "zod";
import { AnswerRelevancyMetric } from "@/metrics/answer-relevancy/answer-relevancy";
import { generateWithSchema } from "@/metrics/utils";
import {
  calculateWeightedSummedScore,
  evaluateGEvalPrompt,
} from "@/metrics/g-eval/utils";
import { DeepEvalBaseLLM } from "@/models";
import type {
  ContentTokenLogProbs,
  GenerationResult,
  RawGenerationOptions,
  RawGenerationResult,
} from "@/models";

const Schema = z.object({ statements: z.array(z.string()) });

/** Returns whatever it is told to, so each contract case can be posed exactly. */
class StubModel extends DeepEvalBaseLLM {
  constructor(
    private readonly result: { output: unknown; cost: number | null },
    private readonly raw?: (
      prompt: string,
      options?: RawGenerationOptions,
    ) => Promise<RawGenerationResult>,
  ) {
    super("stub-model");
    if (raw) {
      this.generateRaw = (prompt, options) => raw(prompt, options);
    }
  }

  async generate<T = string>(): Promise<GenerationResult<T>> {
    return this.result as GenerationResult<T>;
  }

  getModelName(): string {
    return "stub-model";
  }

  // No registry namespace, so nothing is known; GEval's gate needs a `true`.
  supportsLogProbs(): boolean {
    return this.raw !== undefined;
  }
}

function metricWith(model: DeepEvalBaseLLM): AnswerRelevancyMetric {
  const metric = new AnswerRelevancyMetric({ model });
  metric.evaluationCost = 0;
  return metric;
}

describe("generateWithSchema", () => {
  it("passes a model's already-parsed output straight through", async () => {
    const metric = metricWith(
      new StubModel({ output: { statements: ["a"] }, cost: 0 }),
    );
    await expect(generateWithSchema(metric, "p", Schema)).resolves.toEqual({
      statements: ["a"],
    });
  });

  it("recovers when a custom model ignores the schema and returns raw text", async () => {
    const metric = metricWith(
      new StubModel({
        output: 'Sure! {"statements": ["a", "b"]}',
        cost: 0,
      }),
    );
    await expect(generateWithSchema(metric, "p", Schema)).resolves.toEqual({
      statements: ["a", "b"],
    });
  });

  it("explains the schema contract when the output can't be salvaged", async () => {
    const metric = metricWith(
      new StubModel({ output: "no json here", cost: 0 }),
    );
    await expect(generateWithSchema(metric, "p", Schema)).rejects.toThrow(
      /must honor its `schema` argument/,
    );
  });
});

describe("cost accrual", () => {
  it("adds up costs the model reports", async () => {
    const metric = metricWith(
      new StubModel({ output: { statements: [] }, cost: 0.25 }),
    );
    await generateWithSchema(metric, "p", Schema);
    await generateWithSchema(metric, "p", Schema);
    expect(metric.evaluationCost).toBe(0.5);
  });

  it("reports unknown rather than free once a cost is unpriced", async () => {
    const metric = metricWith(
      new StubModel({ output: { statements: [] }, cost: null }),
    );
    await generateWithSchema(metric, "p", Schema);
    expect(metric.evaluationCost).toBeUndefined();
  });

  it("stays unknown even if a later call does report a cost", () => {
    const metric = metricWith(
      new StubModel({ output: { statements: [] }, cost: null }),
    );
    metric.accrueCost(null);
    metric.accrueCost(0.5);
    expect(metric.evaluationCost).toBeUndefined();
  });
});

/** `logprob = ln(p)`, so callers can write probabilities directly. */
function tokenLogProbs(
  token: string,
  alternatives: Record<string, number>,
): ContentTokenLogProbs {
  return {
    token,
    logprob: Math.log(alternatives[token] ?? 1),
    topLogProbs: Object.entries(alternatives).map(([t, p]) => ({
      token: t,
      logprob: Math.log(p),
    })),
  };
}

describe("calculateWeightedSummedScore", () => {
  it("weights the score by the probability of each candidate", () => {
    const logProbs = [tokenLogProbs("8", { "8": 0.7, "7": 0.3 })];
    // (8*0.7 + 7*0.3) / 1.0
    expect(calculateWeightedSummedScore(8, logProbs)).toBeCloseTo(7.7, 10);
  });

  it("renormalizes after dropping candidates under 1% probability", () => {
    const logProbs = [tokenLogProbs("9", { "9": 0.8, "8": 0.2, "1": 0.001 })];
    expect(calculateWeightedSummedScore(9, logProbs)).toBeCloseTo(8.8, 10);
  });

  it("ignores non-decimal candidates", () => {
    const logProbs = [
      tokenLogProbs("5", { "5": 0.6, " ": 0.3, "4": 0.4, four: 0.2 }),
    ];
    // (5*0.6 + 4*0.4) / 1.0
    expect(calculateWeightedSummedScore(5, logProbs)).toBeCloseTo(4.6, 10);
  });

  it("keeps the raw score when nothing survives the filters", () => {
    const logProbs = [tokenLogProbs("3", { "3": 0.001, x: 0.002 })];
    expect(calculateWeightedSummedScore(3, logProbs)).toBe(3);
  });

  it("keeps the raw score when no token matches it", () => {
    const logProbs = [tokenLogProbs("2", { "2": 0.9 })];
    expect(calculateWeightedSummedScore(7, logProbs)).toBe(7);
  });

  it("keeps the raw score when the model reported no log probs", () => {
    expect(calculateWeightedSummedScore(6, undefined)).toBe(6);
  });
});

describe("evaluateGEvalPrompt", () => {
  const rawResult = async (): Promise<RawGenerationResult> => ({
    output: '{"score": 8, "reason": "solid"}',
    cost: 0.1,
    logProbs: [tokenLogProbs("8", { "8": 0.7, "7": 0.3 })],
  });

  it("weights the score when the model exposes log probs", async () => {
    const metric = metricWith(
      new StubModel(
        { output: { score: 8, reason: "solid" }, cost: 0 },
        rawResult,
      ),
    );
    const [score, reason] = await evaluateGEvalPrompt(metric, "p", {
      topLogprobs: 20,
      strictMode: false,
    });
    expect(score).toBeCloseTo(7.7, 10);
    expect(reason).toBe("solid");
  });

  it("returns the unweighted score in strict mode", async () => {
    const metric = metricWith(
      new StubModel(
        { output: { score: 8, reason: "solid" }, cost: 0 },
        rawResult,
      ),
    );
    const [score] = await evaluateGEvalPrompt(metric, "p", {
      topLogprobs: 20,
      strictMode: true,
    });
    expect(score).toBe(8);
  });

  it("falls back to the structured call when the raw path throws", async () => {
    const metric = metricWith(
      new StubModel(
        { output: { score: 4, reason: "fallback" }, cost: 0 },
        async () => {
          throw new Error("this provider rejects top_logprobs");
        },
      ),
    );
    const [score, reason] = await evaluateGEvalPrompt(metric, "p", {
      topLogprobs: 20,
      strictMode: false,
    });
    expect(score).toBe(4);
    expect(reason).toBe("fallback");
  });

  it("uses the structured call for a model with no generateRaw at all", async () => {
    const metric = metricWith(
      new StubModel({ output: { score: 6, reason: "plain" }, cost: 0 }),
    );
    const [score, reason] = await evaluateGEvalPrompt(metric, "p", {
      topLogprobs: 20,
      strictMode: false,
    });
    expect(score).toBe(6);
    expect(reason).toBe("plain");
  });

  it("passes the requested topLogprobs to the model", async () => {
    let seen: number | undefined;
    const metric = metricWith(
      new StubModel(
        { output: { score: 8, reason: "solid" }, cost: 0 },
        async (_prompt, options) => {
          seen = options?.topLogprobs;
          return rawResult();
        },
      ),
    );
    await evaluateGEvalPrompt(metric, "p", {
      topLogprobs: 5,
      strictMode: false,
    });
    expect(seen).toBe(5);
  });
});
