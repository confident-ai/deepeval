// GEval weights the raw score with the log-probs of the *last* token that
// matches it. The prompt asks for `"reason"` before `"score"`, so a digit that
// the reason repeats (a time, a count, an amount) must not be mistaken for the
// score — that is the bug fixed in Python 4.1.9 and ported here.

import type { ContentTokenLogProbs } from "@/models";
import { calculateWeightedSummedScore } from "@/metrics/g-eval/utils";

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

describe("calculateWeightedSummedScore", () => {
  it("weights the score token, not a digit repeated in the reason", () => {
    // The reason mentions "$9,400"; the score is 9, uncertain between 9 and 8.
    const logProbs = [
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

    expect(calculateWeightedSummedScore(9, logProbs)).toBeCloseTo(8.6, 10);
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
