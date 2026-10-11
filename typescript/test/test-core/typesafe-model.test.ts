// TypeSafeModel's translation to and from `@typesafe-ai/sdk`, against a
// virtual mock of the SDK so the suite needs neither the package nor a key.

const systemOne = jest.fn();
const clientOptions: unknown[] = [];

jest.mock(
  "@typesafe-ai/sdk",
  () => ({
    TypeSafeClient: class {
      constructor(options: unknown) {
        clientOptions.push(options);
      }
      systemOne = systemOne;
    },
  }),
  { virtual: true },
);

import { DeepEvalError } from "@/errors";
import { SystemOneContextLimitError, TypeSafeModel } from "@/models/system-one";

describe("TypeSafeModel", () => {
  beforeEach(() => {
    systemOne.mockReset();
    clientOptions.length = 0;
  });

  it("translates questions and answers, and prices input tokens", async () => {
    systemOne.mockResolvedValue({
      model: "jev-latest",
      answers: {
        a: { type: "noul", noul: 0.8 },
        b: {
          type: "choice",
          choice: "billing",
          probabilities: { billing: 0.9, other: 0.1 },
          confidence: 0.8,
        },
        c: {
          type: "score",
          score: 1.5,
          probabilities: { "0": 0.1, "1": 0.3, "2": 0.6 },
          confidence: 0.4,
        },
      },
      usage: { input_tokens: 1000, output_tokens: 20 },
    });
    const model = new TypeSafeModel({ apiKey: "ts-test" });
    const { answers, cost } = await model.decide(
      { ticket: "charged twice" },
      {
        a: { type: "noul", instructions: "Asks for a refund?" },
        b: {
          type: "choice",
          instructions: "Which team?",
          options: { billing: null, other: null },
        },
        c: {
          type: "score",
          instructions: "How urgent?",
          levels: ["l", "m", "h"],
        },
      },
    );

    expect(clientOptions).toEqual([{ apiKey: "ts-test" }]);
    expect(systemOne).toHaveBeenCalledWith({
      model: "jev-latest",
      state: { ticket: "charged twice" },
      questions: {
        a: { type: "noul", instructions: "Asks for a refund?" },
        b: {
          type: "choice",
          instructions: "Which team?",
          criteria: { billing: null, other: null },
        },
        c: {
          type: "score",
          instructions: "How urgent?",
          criteria: ["l", "m", "h"],
        },
      },
    });
    expect(answers.nouls.a.probability).toBe(0.8);
    expect(answers.nouls.a.confidence).toBeCloseTo(0.6);
    expect(answers.choices.b.choice).toBe("billing");
    expect(answers.scores.c.probabilities).toEqual({ 0: 0.1, 1: 0.3, 2: 0.6 });
    expect(answers.scores.c.normalized).toBeCloseTo(0.75);
    expect(cost).toBeCloseTo(1000 * (0.042 / 1e6));
    expect(model.getModelName()).toBe("jev-latest (TypeSafe AI)");
  });

  it("sends Noul criteria only when given", async () => {
    systemOne.mockResolvedValue({
      answers: { a: { type: "noul", noul: 0.5 } },
    });
    const model = new TypeSafeModel({ apiKey: "ts-test", model: "jev-1.13.0" });
    await model.decide("s", {
      a: { type: "noul", instructions: "i", true: "holds", false: undefined },
    });
    expect(systemOne.mock.calls[0][0].questions.a).toEqual({
      type: "noul",
      instructions: "i",
      criteria: { true: "holds", false: null },
    });
    expect(systemOne.mock.calls[0][0].model).toBe("jev-1.13.0");
  });

  it("leaves an unknown model unpriced unless a cost is given", async () => {
    systemOne.mockResolvedValue({ answers: {}, usage: { input_tokens: 10 } });
    const unpriced = new TypeSafeModel({ apiKey: "k", model: "jev-custom" });
    expect(
      (await unpriced.decide("s", { a: { type: "noul", instructions: "i" } }))
        .cost,
    ).toBeNull();
    const priced = new TypeSafeModel({
      apiKey: "k",
      model: "jev-custom",
      costPerInputToken: 0.5,
    });
    expect(
      (await priced.decide("s", { a: { type: "noul", instructions: "i" } }))
        .cost,
    ).toBe(5);
  });

  it("fails at construction without an API key", () => {
    const saved = process.env.TYPESAFE_API_KEY;
    delete process.env.TYPESAFE_API_KEY;
    try {
      expect(() => new TypeSafeModel()).toThrow(/TYPESAFE_API_KEY/);
    } finally {
      if (saved !== undefined) process.env.TYPESAFE_API_KEY = saved;
    }
  });

  it("refuses an empty question set", async () => {
    const model = new TypeSafeModel({ apiKey: "k" });
    await expect(model.decide("s", {})).rejects.toThrow(DeepEvalError);
    expect(systemOne).not.toHaveBeenCalled();
  });

  it("checks the context budget before calling the API", async () => {
    const model = new TypeSafeModel({ apiKey: "k" });
    await expect(
      model.decide("x".repeat(200_000), {
        a: { type: "noul", instructions: "i" },
      }),
    ).rejects.toThrow(SystemOneContextLimitError);
    expect(systemOne).not.toHaveBeenCalled();
  });
});
