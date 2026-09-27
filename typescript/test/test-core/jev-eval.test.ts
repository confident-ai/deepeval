// JevEval and ConversationalJevEval: Jev-native score metrics. Every test runs
// against a fake System One model, so no network, API key or LLM is needed:
// neither metric ever calls one.

import { Choice, ConversationalJevEval, JevEval, Noul, Score } from "@/metrics";
import {
  aggregate,
  buildQuestions,
  constructMultiTurnState,
  constructSingleTurnState,
  outcomesFromAnswers,
  verbaliseOutcome,
} from "@/metrics/jev-eval/utils";
import {
  ChoiceAnswer,
  NoulAnswer,
  ScoreAnswer,
  SystemOneAnswers,
} from "@/models/system-one";
import {
  ConversationalTestCase,
  LLMTestCase,
  MultiTurnParams,
  SingleTurnParams,
  ToolCall,
  Turn,
} from "@/test-case";
import { FakeSystemOneModel } from "./system-one-fakes";

// The worked example from the docs: Tool Faithfulness.
const QUESTIONS = [
  new Noul({
    statement:
      "Every fact and figure in actual_output appears in the output of a tool in tools_called.",
    weight: 2,
  }),
  new Noul({
    statement:
      "actual_output reports every value returned in tools_called accurately.",
  }),
  new Score({
    question:
      "How much of actual_output is grounded in the outputs in tools_called?",
    levels: [
      "Fabricated",
      "Mostly fabricated",
      "Mostly grounded",
      "Fully grounded",
    ],
  }),
  new Choice({
    question:
      "What did actual_output do with information the tools did not return?",
    options: {
      left_it_out: 1.0,
      flagged_it_as_unknown: 1.0,
      hedged_it: 0.5,
      stated_it_as_fact: 0.0,
      nothing_missing: null,
    },
  }),
];

const TOOLS = [
  new ToolCall({
    name: "get_weather",
    inputParameters: { city: "Paris" },
    output: { temp_c: 18, condition: "sunny" },
  }),
];

const TEST_CASE = new LLMTestCase({
  input: "What's the weather in Paris right now?",
  actualOutput:
    "It's 18°C and sunny in Paris, with a light breeze and around 40% humidity.",
  toolsCalled: TOOLS,
});

function exampleAnswers(choice?: ChoiceAnswer): SystemOneAnswers {
  return new SystemOneAnswers({
    nouls: { q_0: new NoulAnswer(0.3), q_1: new NoulAnswer(0.9) },
    scores: {
      q_2: new ScoreAnswer(1.7, { 0: 0.05, 1: 0.3, 2: 0.55, 3: 0.1 }, 0.55),
    },
    choices: {
      q_3:
        choice ??
        new ChoiceAnswer(
          "stated_it_as_fact",
          {
            left_it_out: 0.05,
            flagged_it_as_unknown: 0.05,
            hedged_it: 0.25,
            stated_it_as_fact: 0.6,
            nothing_missing: 0.05,
          },
          0.6,
        ),
    },
  });
}

const NOT_APPLICABLE_CHOICE = new ChoiceAnswer(
  "nothing_missing",
  {
    left_it_out: 0.04,
    flagged_it_as_unknown: 0.02,
    hedged_it: 0.02,
    stated_it_as_fact: 0.02,
    nothing_missing: 0.9,
  },
  0.9,
);

function makeMetric(
  answers: SystemOneAnswers = exampleAnswers(),
  options: Partial<ConstructorParameters<typeof JevEval>[0]> = {},
): JevEval {
  return new JevEval({
    name: "Tool Faithfulness",
    evaluationParams: [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.TOOLS_CALLED,
    ],
    questions: QUESTIONS,
    systemOneModel: new FakeSystemOneModel({ answers }),
    includeReason: false,
    showIndicator: false,
    ...options,
  });
}

function fakeOf(metric: { systemOneModel?: unknown }): FakeSystemOneModel {
  return metric.systemOneModel as FakeSystemOneModel;
}

describe("JevEval value mapping", () => {
  it("scores the worked example as a weighted mean", async () => {
    const metric = makeMetric();
    const score = await metric.measure(TEST_CASE);
    const expected = (2 * 0.3 + 0.9 + 1.7 / 3 + 0.225 / 0.95) / 5;
    expect(score).toBeCloseTo(expected, 9);
    expect(score).toBeCloseTo(0.461, 3);
    expect(metric.success).toBe(false);
    expect(metric.reason).toBeUndefined();
    // Least decisive: the first Noul at P=0.30 -> |2*0.30 - 1| = 0.40.
    expect(metric.confidence).toBeCloseTo(0.4);
  });

  it("maps each answer onto [0, 1]", () => {
    const outcomes = outcomesFromAnswers(QUESTIONS, exampleAnswers());
    expect(outcomes.map((o) => o.type)).toEqual([
      "noul",
      "noul",
      "score",
      "choice",
    ]);
    expect(outcomes[0].value).toBeCloseTo(0.3);
    expect(outcomes[0].confidence).toBeCloseTo(0.4);
    expect(outcomes[1].confidence).toBeCloseTo(0.8);
    expect(outcomes[0].weight).toBe(2);
    expect(outcomes[0].probabilities.true).toBeCloseTo(0.3);
    expect(outcomes[0].probabilities.false).toBeCloseTo(0.7);
    expect(outcomes[2].value).toBeCloseTo(1.7 / 3);
    expect(outcomes[2].probabilities["Mostly grounded"]).toBeCloseTo(0.55);
    expect(outcomes[3].value).toBeCloseTo(0.225 / 0.95);
    expect(outcomes.every((o) => o.applicable)).toBe(true);
  });

  it("drops a Choice whose not-applicable mass dominates", async () => {
    const answers = exampleAnswers(NOT_APPLICABLE_CHOICE);
    const metric = makeMetric(answers);
    const score = await metric.measure(TEST_CASE);
    const outcomes = outcomesFromAnswers(QUESTIONS, answers);
    expect(outcomes[3].applicable).toBe(false);
    expect(outcomes[3].value).toBeNull();
    expect(score).toBeCloseTo((2 * 0.3 + 0.9 + 1.7 / 3) / 4);
    expect((metric.scoreBreakdown as any[])[3].applicable).toBe(false);
  });

  it("excludes a Choice exactly at the not-applicable threshold", () => {
    const q = new Choice({ question: "q", options: { a: 1, na: null } });
    const answers = new SystemOneAnswers({
      choices: { q_0: new ChoiceAnswer("na", { a: 0.5, na: 0.5 }, 0) },
    });
    expect(outcomesFromAnswers([q], answers)[0].applicable).toBe(false);
  });

  it("handles non-monotone Choice credits", () => {
    const q = new Choice({
      question: "how did it comply?",
      options: { complied_unsafely: 0, declined: 0, complied_safely: 1 },
    });
    const answers = new SystemOneAnswers({
      choices: {
        q_0: new ChoiceAnswer(
          "complied_safely",
          { complied_unsafely: 0.2, declined: 0.2, complied_safely: 0.6 },
          0.6,
        ),
      },
    });
    expect(outcomesFromAnswers([q], answers)[0].value).toBeCloseTo(0.6);
  });

  it("scores 1 when nothing applies", () => {
    const q = new Choice({ question: "q", options: { a: 1, na: null } });
    const answers = new SystemOneAnswers({
      choices: { q_0: new ChoiceAnswer("na", { a: 0.1, na: 0.9 }, 0.9) },
    });
    expect(aggregate(outcomesFromAnswers([q], answers))).toBe(1);
  });

  it("weights change the mean", () => {
    const a = new Noul({ statement: "a", weight: 3 });
    const b = new Noul({ statement: "b", weight: 1 });
    const answers = new SystemOneAnswers({
      nouls: { q_0: new NoulAnswer(1), q_1: new NoulAnswer(0) },
    });
    expect(aggregate(outcomesFromAnswers([a, b], answers))).toBeCloseTo(0.75);
  });

  it("maps a two-level Score onto its top index", () => {
    const q = new Score({ question: "q", levels: ["bad", "good"] });
    const answers = new SystemOneAnswers({
      scores: { q_0: new ScoreAnswer(0.8, { 0: 0.2, 1: 0.8 }, 0.8) },
    });
    expect(outcomesFromAnswers([q], answers)[0].value).toBeCloseTo(0.8);
  });
});

describe("JevEval strict mode", () => {
  it("forces threshold 1 and a binary score", async () => {
    const metric = makeMetric(undefined, { strictMode: true });
    expect(metric.threshold).toBe(1);
    expect(await metric.measure(TEST_CASE)).toBe(0);
    expect(metric.success).toBe(false);
    expect((metric.scoreBreakdown as any[]).map((o) => o.passed)).toEqual([
      false,
      true,
      false,
      false,
    ]);
  });

  it("scores 1 when every applicable answer is at its best", async () => {
    const answers = new SystemOneAnswers({
      nouls: { q_0: new NoulAnswer(0.9), q_1: new NoulAnswer(0.7) },
      scores: {
        q_2: new ScoreAnswer(2.8, { 0: 0, 1: 0.05, 2: 0.1, 3: 0.85 }, 0.85),
      },
      choices: {
        q_3: new ChoiceAnswer(
          "left_it_out",
          {
            left_it_out: 0.7,
            flagged_it_as_unknown: 0.2,
            hedged_it: 0.05,
            stated_it_as_fact: 0.03,
            nothing_missing: 0.02,
          },
          0.7,
        ),
      },
    });
    const metric = makeMetric(answers, { strictMode: true });
    expect(await metric.measure(TEST_CASE)).toBe(1);
    expect(metric.success).toBe(true);
  });

  it("skips a not-applicable Choice", async () => {
    const answers = new SystemOneAnswers({
      nouls: { q_0: new NoulAnswer(0.9), q_1: new NoulAnswer(0.9) },
      scores: {
        q_2: new ScoreAnswer(2.9, { 0: 0, 1: 0, 2: 0.1, 3: 0.9 }, 0.9),
      },
      choices: { q_3: NOT_APPLICABLE_CHOICE },
    });
    const metric = makeMetric(answers, { strictMode: true });
    expect(await metric.measure(TEST_CASE)).toBe(1);
    const breakdown = metric.scoreBreakdown as any[];
    expect(breakdown[3].applicable).toBe(false);
    expect(breakdown[3].passed).toBeNull();
  });

  it("leaves `passed` unset outside strict mode", async () => {
    const metric = makeMetric();
    await metric.measure(TEST_CASE);
    expect(
      (metric.scoreBreakdown as any[]).every((o) => o.passed === null),
    ).toBe(true);
  });
});

describe("JevEval request shape", () => {
  it("sends only the evaluation params, under their Python names", () => {
    const state = constructSingleTurnState(
      [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
      TEST_CASE,
    );
    expect(state).toEqual({
      test_case: {
        input: TEST_CASE.input,
        actual_output: TEST_CASE.actualOutput,
      },
    });
  });

  it("keeps credits on the DeepEval side", () => {
    const questions = buildQuestions(QUESTIONS);
    expect(Object.keys(questions)).toEqual(["q_0", "q_1", "q_2", "q_3"]);
    expect(questions.q_0.type).toBe("noul");
    expect(questions.q_2).toMatchObject({
      type: "score",
      levels: QUESTIONS[2] instanceof Score ? QUESTIONS[2].levels : [],
    });
    expect(questions.q_3).toEqual({
      type: "choice",
      instructions: QUESTIONS[3].text,
      options: {
        left_it_out: null,
        flagged_it_as_unknown: null,
        hedged_it: null,
        stated_it_as_fact: null,
        nothing_missing: null,
      },
    });
  });

  it("structures tool calls with snake_case keys", () => {
    const state = constructSingleTurnState(
      [SingleTurnParams.TOOLS_CALLED],
      TEST_CASE,
    );
    expect(state.test_case.tools_called).toEqual([
      {
        name: "get_weather",
        input_parameters: { city: "Paris" },
        output: { temp_c: 18, condition: "sunny" },
      },
    ]);
  });

  it("makes one decide call per measure", async () => {
    const metric = makeMetric();
    await metric.measure(TEST_CASE);
    const fake = fakeOf(metric);
    expect(fake.calls).toHaveLength(1);
    expect(Object.keys(fake.calls[0].state.test_case).sort()).toEqual([
      "actual_output",
      "input",
      "tools_called",
    ]);
    expect(Object.keys(fake.calls[0].questions)).toHaveLength(4);
  });

  it("reports the System One model as the evaluation model", async () => {
    const metric = makeMetric();
    await metric.measure(TEST_CASE);
    expect(metric.reason).toBeUndefined();
    expect(metric.evaluationModel).toBe("fake-jev");
    expect(metric.model).toBeUndefined();
  });
});

describe("JevEval reason", () => {
  it("is deterministic and covers every question", async () => {
    const metric = makeMetric(undefined, { includeReason: true });
    await metric.measure(TEST_CASE);
    const reason = metric.reason as string;
    expect(
      reason.startsWith("Decided by fake-jev, minimum confidence 0.40."),
    ).toBe(true);
    for (const q of QUESTIONS) expect(reason).toContain(q.text);
    expect(reason).toContain(
      "likely fails (P(yes)=0.30, confidence=0.40, weight=2)",
    );
    expect(reason).toContain("clearly holds (P(yes)=0.90, confidence=0.80)");
    expect(reason).toContain('"Mostly grounded"');
    expect(reason).toContain("expected level=0.57 of 1.00");
    expect(reason).toContain('"stated_it_as_fact" (P=0.60, confidence=0.60)');
    expect(
      reason.endsWith("Score: 0.46 (weighted mean of 4 applicable questions)."),
    ).toBe(true);

    const again = makeMetric(undefined, { includeReason: true });
    await again.measure(TEST_CASE);
    expect(again.reason).toBe(reason);
  });

  it("marks strict results", async () => {
    const metric = makeMetric(undefined, {
      includeReason: true,
      strictMode: true,
    });
    await metric.measure(TEST_CASE);
    expect(metric.reason).toContain("strict=fail");
    expect(metric.reason).toContain("strict=pass");
    expect(
      metric.reason?.endsWith(
        "Score: 0.00 (strict mode: at least one applicable question failed).",
      ),
    ).toBe(true);
  });

  it("marks a not-applicable Choice", async () => {
    const metric = makeMetric(exampleAnswers(NOT_APPLICABLE_CHOICE), {
      includeReason: true,
    });
    await metric.measure(TEST_CASE);
    expect(metric.reason).toContain("-> not applicable (not applicable");
    expect(metric.reason).toContain("weighted mean of 3 applicable questions");
  });

  it("verbalises Noul bands", () => {
    const noul = (p: number) =>
      outcomesFromAnswers(
        [new Noul({ statement: "x" })],
        new SystemOneAnswers({ nouls: { q_0: new NoulAnswer(p) } }),
      )[0];
    expect(verbaliseOutcome(noul(0.9))).toBe("clearly holds");
    expect(verbaliseOutcome(noul(0.7))).toBe("likely holds");
    expect(verbaliseOutcome(noul(0.5))).toBe("unclear");
    expect(verbaliseOutcome(noul(0.2))).toBe("likely fails");
    expect(verbaliseOutcome(noul(0.05))).toBe("clearly fails");
  });
});

describe("JevEval validation", () => {
  it("rejects a multimodal test case before calling Jev", async () => {
    const metric = makeMetric();
    await expect(
      metric.measure(
        new LLMTestCase({
          input: "what is this?",
          actualOutput: "a cat",
          toolsCalled: TOOLS,
          multimodal: true,
        }),
      ),
    ).rejects.toThrow(/text only/);
    expect(fakeOf(metric).calls).toEqual([]);
  });

  it("rejects empty questions", () => {
    expect(() => makeMetric(undefined, { questions: [] })).toThrow(
      /at least one question/,
    );
  });

  it("rejects missing evaluation params", () => {
    expect(() => makeMetric(undefined, { evaluationParams: [] })).toThrow(
      /evaluationParams cannot be empty/,
    );
  });

  it("rejects a non-question", () => {
    expect(() =>
      makeMetric(undefined, { questions: ["not a question"] as any }),
    ).toThrow(TypeError);
  });

  it.each([1, 11])("rejects a Score with %i levels", (n) => {
    expect(
      () =>
        new Score({
          question: "q",
          levels: Array.from({ length: n }, (_, i) => `l${i}`),
        }),
    ).toThrow(/between 2 and 10/);
  });

  it("rejects duplicate Score levels", () => {
    expect(() => new Score({ question: "q", levels: ["a", "a"] })).toThrow(
      /unique/,
    );
  });

  it("rejects out-of-range Choice credits", () => {
    expect(
      () => new Choice({ question: "q", options: { a: 1.5, b: 0 } }),
    ).toThrow(/between 0 and 1/);
  });

  it("rejects a Choice with no credited option", () => {
    expect(
      () => new Choice({ question: "q", options: { a: null, b: null } }),
    ).toThrow(/at least one option with a credit/);
  });

  it("rejects a Choice with one option", () => {
    expect(() => new Choice({ question: "q", options: { a: 1 } })).toThrow(
      /at least 2 options/,
    );
  });

  it("rejects a non-positive weight", () => {
    expect(() => new Noul({ statement: "a", weight: 0 })).toThrow(/weight/);
  });

  it("suffixes the name", () => {
    expect(makeMetric().name).toBe("Tool Faithfulness [JevEval]");
    expect(makeMetric(undefined, { includeJevEvalSuffix: false }).name).toBe(
      "Tool Faithfulness",
    );
  });

  it("raises on a missing test case param", async () => {
    await expect(
      makeMetric().measure(
        new LLMTestCase({ input: "hi", actualOutput: "there" }),
      ),
    ).rejects.toThrow();
  });
});

// ConversationalJevEval

const CONVERSATION = new ConversationalTestCase({
  scenario: "A customer asks for a refund on a late order.",
  expectedOutcome: "The refund amount is confirmed.",
  turns: [
    new Turn({
      role: "user",
      content: "My order arrived a week late. I want a refund.",
    }),
    new Turn({
      role: "assistant",
      content: "Sorry about that. What is your order number?",
    }),
    new Turn({ role: "user", content: "It's 4471. How much do I get back?" }),
    new Turn({
      role: "assistant",
      content: "You'll receive a refund within 5-7 days.",
    }),
    new Turn({ role: "user", content: "ok thanks" }),
  ],
});

const CONVERSATION_QUESTIONS = [
  new Noul({
    statement: "The assistant verifies the order before acting on the refund.",
  }),
  new Score({
    question: "How fully is the user's request settled by the end of turns?",
    levels: ["Not addressed", "Partly settled", "Fully resolved"],
  }),
  new Choice({
    question: "What did the assistant do about the amount the user asked for?",
    options: {
      stated_it: 1,
      deferred_it: 0.5,
      ignored_it: 0,
      never_asked: null,
    },
  }),
];

const CONVERSATION_ANSWERS = new SystemOneAnswers({
  nouls: { q_0: new NoulAnswer(0.9) },
  scores: { q_1: new ScoreAnswer(1.1, { 0: 0.1, 1: 0.7, 2: 0.2 }, 0.7) },
  choices: {
    q_2: new ChoiceAnswer(
      "ignored_it",
      {
        stated_it: 0.05,
        deferred_it: 0.25,
        ignored_it: 0.65,
        never_asked: 0.05,
      },
      0.65,
    ),
  },
});

function makeConversationalMetric(
  options: Partial<ConstructorParameters<typeof ConversationalJevEval>[0]> = {},
): ConversationalJevEval {
  return new ConversationalJevEval({
    name: "Refund Handling",
    evaluationParams: [
      MultiTurnParams.SCENARIO,
      MultiTurnParams.EXPECTED_OUTCOME,
    ],
    questions: CONVERSATION_QUESTIONS,
    systemOneModel: new FakeSystemOneModel({ answers: CONVERSATION_ANSWERS }),
    includeReason: false,
    showIndicator: false,
    ...options,
  });
}

describe("ConversationalJevEval", () => {
  it("always includes turn content and role", () => {
    const metric = makeConversationalMetric();
    expect(metric.evaluationParams).toContain(MultiTurnParams.CONTENT);
    expect(metric.evaluationParams).toContain(MultiTurnParams.ROLE);
  });

  it("builds turns plus conversation-level fields", () => {
    const state = constructMultiTurnState(
      [
        MultiTurnParams.SCENARIO,
        MultiTurnParams.EXPECTED_OUTCOME,
        MultiTurnParams.CONTENT,
        MultiTurnParams.ROLE,
      ],
      CONVERSATION,
    );
    expect(Object.keys(state).sort()).toEqual(["test_case", "turns"]);
    expect(state.turns).toHaveLength(5);
    expect(state.turns[1]).toEqual({
      content: "Sorry about that. What is your order number?",
      role: "assistant",
    });
    expect(state.test_case).toEqual({
      scenario: CONVERSATION.scenario,
      expected_outcome: CONVERSATION.expectedOutcome,
    });
  });

  it("measures end to end", async () => {
    const metric = makeConversationalMetric();
    const score = await metric.measure(CONVERSATION);
    const choiceValue = (0.05 * 1 + 0.25 * 0.5) / 0.95;
    expect(score).toBeCloseTo((0.9 + 0.55 + choiceValue) / 3);
    expect(metric.success).toBe(true);
    expect(metric.scoreBreakdown).toHaveLength(3);
    expect(fakeOf(metric).calls[0].state.turns[0].role).toBe("user");
    expect(metric.name).toBe("Refund Handling [Conversational JevEval]");
    expect(metric.confidence).toBeCloseTo(0.65);
    expect(metric.evaluationModel).toBe("fake-jev");
  });

  it("gives a deterministic reason citing every question", async () => {
    const metric = makeConversationalMetric({ includeReason: true });
    await metric.measure(CONVERSATION);
    const reason = metric.reason as string;
    expect(
      reason.startsWith("Decided by fake-jev, minimum confidence 0.65."),
    ).toBe(true);
    for (const q of CONVERSATION_QUESTIONS) expect(reason).toContain(q.text);
    expect(reason).toContain("clearly holds (P(yes)=0.90, confidence=0.80)");
    expect(reason).toContain('"Partly settled"');
    expect(reason).toContain('"ignored_it" (P=0.65, confidence=0.65)');
    expect(
      reason.endsWith("Score: 0.54 (weighted mean of 3 applicable questions)."),
    ).toBe(true);
  });

  it("raises on a missing scenario", async () => {
    await expect(
      makeConversationalMetric().measure(
        new ConversationalTestCase({
          turns: [new Turn({ role: "user", content: "hi" })],
        }),
      ),
    ).rejects.toThrow();
  });
});
