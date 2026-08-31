// Rubric score ranges on any scale — 0-10, 0-1 with decimals, 1-5, 0-100.
// Bounds are arbitrary finite numbers and the metric normalizes the judge's
// score to 0-1 by the rubric's own span. Integral scales must keep rendering
// exactly as they did when the range was hard-coded to 0-10, since prompt drift
// silently changes eval results.

import {
  type Rubric,
  formatRubrics,
  formatScoreRange,
  getScoreRange,
  isIntegralRubricScale,
  normalizeScore,
  validateAndSortRubrics,
} from "@/metrics/g-eval/utils";

const FRACTIONAL_RUBRIC: Rubric[] = [
  { scoreRange: [0.0, 0.3], expectedOutcome: "Poor" },
  { scoreRange: [0.4, 0.7], expectedOutcome: "OK" },
  { scoreRange: [0.8, 1.0], expectedOutcome: "Great" },
];

const INTEGER_RUBRIC: Rubric[] = [
  { scoreRange: [0, 5], expectedOutcome: "Nice" },
  { scoreRange: [6, 10], expectedOutcome: "Not so Nice" },
];

describe("validateAndSortRubrics", () => {
  it("sorts fractional bands by start", () => {
    const sorted = validateAndSortRubrics([
      FRACTIONAL_RUBRIC[2],
      FRACTIONAL_RUBRIC[0],
      FRACTIONAL_RUBRIC[1],
    ]);
    expect(sorted?.map((r) => r.expectedOutcome)).toEqual([
      "Poor",
      "OK",
      "Great",
    ]);
  });

  it("still rejects bands that touch", () => {
    expect(() =>
      validateAndSortRubrics([
        { scoreRange: [0.0, 0.5], expectedOutcome: "Poor" },
        { scoreRange: [0.5, 1.0], expectedOutcome: "Great" },
      ]),
    ).toThrow(/Overlapping score ranges/);
  });

  it("rejects an inverted range", () => {
    expect(() =>
      validateAndSortRubrics([
        { scoreRange: [0.6, 0.5], expectedOutcome: "Poor" },
      ]),
    ).toThrow(/less than or equal to end/);
  });

  it("rejects a non-finite bound", () => {
    expect(() =>
      validateAndSortRubrics([
        { scoreRange: [0, Number.POSITIVE_INFINITY], expectedOutcome: "Poor" },
      ]),
    ).toThrow(/finite numbers/);
  });
});

describe("isIntegralRubricScale", () => {
  it("treats no rubric as the integral 0-10 default", () => {
    expect(isIntegralRubricScale(undefined)).toBe(true);
  });

  it("recognizes an integer rubric", () => {
    expect(isIntegralRubricScale(INTEGER_RUBRIC)).toBe(true);
  });

  it("looks at every band, not just the outer bounds", () => {
    // 0.0 and 1.0 are whole numbers, but 0.3/0.4 make this a decimal scale.
    expect(getScoreRange(FRACTIONAL_RUBRIC)).toEqual([0, 1]);
    expect(isIntegralRubricScale(FRACTIONAL_RUBRIC)).toBe(false);
  });
});

describe("formatScoreRange", () => {
  it("leaves the default range bare", () => {
    expect(formatScoreRange(undefined)).toEqual(["0", "10"]);
  });

  it("leaves an integer rubric bare", () => {
    expect(formatScoreRange(INTEGER_RUBRIC)).toEqual(["0", "10"]);
  });

  it("spells out the tenth on a decimal scale", () => {
    // `${1.0}` is "1" in JS, which next to "0.0-0.3" reads as another scale.
    expect(formatScoreRange(FRACTIONAL_RUBRIC)).toEqual(["0.0", "1.0"]);
  });
});

describe("normalizeScore", () => {
  it.each<[number, [number, number], number]>([
    [0.65, [0, 1], 0.65],
    [7, [0, 10], 0.7],
    [3, [1, 5], 0.5],
    [50, [0, 100], 0.5],
    [0, [0, 10], 0],
    [10, [0, 10], 1],
  ])("maps %p on %p onto %p", (rawScore, scoreRange, expected) => {
    expect(normalizeScore(rawScore, scoreRange)).toBeCloseTo(expected, 10);
  });

  it.each([
    [1.5, 1],
    [-0.2, 0],
    [11, 1],
  ])("clamps an out-of-range judge score %p to %p", (rawScore, expected) => {
    expect(normalizeScore(rawScore, [0, 1])).toBe(expected);
  });

  it("does not divide by zero on a single-point rubric", () => {
    expect(normalizeScore(1, [1, 1])).toBe(1);
    expect(normalizeScore(0, [1, 1])).toBe(0);
  });
});

describe("formatRubrics", () => {
  it("renders integer bands without decimals", () => {
    expect(formatRubrics(INTEGER_RUBRIC)).toBe("0-5: Nice\n6-10: Not so Nice");
  });

  it("keeps fractional bands decimal", () => {
    expect(formatRubrics(FRACTIONAL_RUBRIC)).toBe(
      "0.0-0.3: Poor\n0.4-0.7: OK\n0.8-1.0: Great",
    );
  });

  it("renders a single-point band once", () => {
    expect(
      formatRubrics([{ scoreRange: [1, 1], expectedOutcome: "yes" }]),
    ).toBe("1: yes");
  });

  it("keeps a whole-numbered band decimal inside a decimal scale", () => {
    expect(
      formatRubrics([
        { scoreRange: [0.0, 0.3], expectedOutcome: "Poor" },
        { scoreRange: [1.0, 1.0], expectedOutcome: "Great" },
      ]),
    ).toBe("0.0-0.3: Poor\n1.0: Great");
  });
});
