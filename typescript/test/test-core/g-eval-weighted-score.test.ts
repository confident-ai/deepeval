// GEval weights the raw score with the log-probs of the *last* token that
// matches it. The prompt asks for `"reason"` before `"score"`, so a digit that
// the reason repeats (a time, a count, an amount) must not be mistaken for the
// score — that is the bug fixed in Python 4.1.9 and ported here.

import type { ContentTokenLogProbs } from "@/models";
import { DeepEvalBaseLLM } from "@/models";
import { GEval } from "@/metrics";
import { calculateWeightedSummedScore } from "@/metrics/g-eval/utils";
import { LLMTestCase, SingleTurnParams } from "@/test-case";

/** One token, with the alternatives the model weighed at that position. */
function token(
  value: string,
  alternatives: Array<[string, number]> = [[value, 1]],
): ContentTokenLogProbs {
  const probability = (candidate: string) =>
    alternatives.find(([alt]) => alt === candidate)?.[1] ?? 0;
  return {
    token: value,
    logprob: Math.log(probability(value)),
    topLogProbs: alternatives.map(([alt, prob]) => ({
      token: alt,
      logprob: Math.log(prob),
    })),
  };
}

/** The reproduction from the issue: the reason mentions "$9,400", the score is 9. */
function scoreAfterRepeatedDigit(): ContentTokenLogProbs[] {
  return [
    token("{"),
    token("reason"),
    token("$"),
    token("9", [
      ["9", 0.999],
      ["4", 0.001],
    ]),
    token(","),
    token("4"),
    token("0"),
    token("0"),
    token("score"),
    token(" "),
    token("9", [
      ["9", 0.6],
      ["8", 0.4],
    ]),
    token("}"),
  ];
}

/** A judge that answers through the raw log-prob path. */
class LogProbJudge extends DeepEvalBaseLLM {
  constructor(private readonly logProbs: ContentTokenLogProbs[]) {
    super("logprob-judge");
  }

  async generate<T = string>(): Promise<{ output: T; cost: number }> {
    // Evaluation steps; the score itself comes from `generateRaw` below.
    return { output: { steps: ["Check the answer."] } as T, cost: 0 };
  }

  async generateRaw(): Promise<{
    output: string;
    cost: number;
    logProbs: ContentTokenLogProbs[];
  }> {
    return {
      output: JSON.stringify({ reason: "It mentions $9,400.", score: 9 }),
      cost: 0,
      logProbs: this.logProbs,
    };
  }

  getModelName(): string {
    return "logprob-judge";
  }
}

describe("calculateWeightedSummedScore", () => {
  it("weights the score token, not a digit repeated in the reason", () => {
    expect(
      calculateWeightedSummedScore(9, scoreAfterRepeatedDigit()),
    ).toBeCloseTo(8.6, 10);
  });

  it("keeps weighing when the score is the only matching token", () => {
    const logProbs = [
      token("score"),
      token(" "),
      token("9", [
        ["9", 0.75],
        ["7", 0.25],
      ]),
    ];

    expect(calculateWeightedSummedScore(9, logProbs)).toBeCloseTo(8.5, 10);
  });

  it("returns the raw score when no token matches", () => {
    const logProbs = [token("score"), token(" "), token("8")];

    expect(calculateWeightedSummedScore(9, logProbs)).toBe(9);
  });

  it("returns the raw score when log-probs are unavailable", () => {
    expect(calculateWeightedSummedScore(9, undefined)).toBe(9);
  });

  it("ignores candidates below the 1% probability floor", () => {
    const logProbs = [
      token("score"),
      token(" "),
      token("9", [
        ["9", 0.004],
        ["8", 0.003],
      ]),
    ];

    expect(calculateWeightedSummedScore(9, logProbs)).toBe(9);
  });

  it("ignores candidates that are not whole numbers", () => {
    const logProbs = [
      token("score"),
      token(" "),
      token("9", [
        ["9", 0.5],
        ["9.5", 0.5],
      ]),
    ];

    expect(calculateWeightedSummedScore(9, logProbs)).toBe(9);
  });
});

describe("GEval.measure", () => {
  it("scores with the score token, not the digit in the reason", async () => {
    const metric = new GEval({
      name: "cost accuracy",
      criteria: "Does the answer quote the cost correctly?",
      evaluationParams: [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
      ],
      model: new LogProbJudge(scoreAfterRepeatedDigit()),
      showIndicator: false,
    });
    const testCase = new LLMTestCase({
      input: "What did it cost?",
      actualOutput: "$9,400",
    });

    // 8.6 out of 10, normalized; the reason's 99.9% digit would give 0.9.
    await expect(metric.measure(testCase)).resolves.toBeCloseTo(0.86, 10);
  });
});
