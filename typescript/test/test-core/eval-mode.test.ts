// Eval modes: who decides in an LLM-as-a-judge metric. A toy metric pins the
// routing contract (llm / hybrid / system_one), then the same contract is
// checked on real metrics: AnswerRelevancy, DAG and GEval.

import { z } from "zod";
import {
  EvalMode,
  resolveEvalMode,
  type EvalModeName,
} from "@/config/eval-mode";
import { DeepEvalError } from "@/errors";
import { BaseMetric } from "@/metrics/base-metrics";
import {
  AnswerRelevancyMetric,
  BinaryJudgementNode,
  DAGMetric,
  DeepAcyclicGraph,
  GEval,
  Noul,
  type JevQuestion,
} from "@/metrics";
import {
  effectiveEvalMode,
  generateQagVerdicts,
  runSystemOneEval,
  splitSentences,
  systemOneActive,
  type SystemOneEvalSpec,
} from "@/metrics/system-one";
import {
  checkSingleTurnParams,
  generateWithSchema,
  initializeMetricModels,
} from "@/metrics/utils";
import { DeepEvalBaseLLM } from "@/models";
import {
  NoulAnswer,
  ScoreAnswer,
  SystemOneAnswers,
  SystemOneContextLimitError,
  type DeepEvalBaseSystemOneModel,
} from "@/models/system-one";
import { LLMTestCase, SingleTurnParams } from "@/test-case";
import {
  ExplodingLLM,
  ExplodingSystemOneModel,
  FakeSystemOneModel,
  ScriptedLLM,
  noulAnswers,
} from "./system-one-fakes";

const ENV_KEYS = ["DEEPEVAL_EVAL_MODE", "DEEPEVAL_MODE", "TYPESAFE_API_KEY"];
const savedEnv: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const key of ENV_KEYS) {
    savedEnv[key] = process.env[key];
    delete process.env[key];
  }
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
});

const TEST_CASE = new LLMTestCase({
  input: "What is the capital of France?",
  actualOutput: "Paris is the capital of France. It has great food.",
});

const TOY_QUESTIONS: JevQuestion[] = [
  new Noul({ statement: "actual_output answers input.", weight: 2 }),
  new Noul({ statement: "actual_output stays on topic." }),
];

/** P=0.95 and P=0.85: least decisive |2*0.85 - 1| = 0.70. */
const CONFIDENT = new SystemOneAnswers({
  nouls: { q_0: new NoulAnswer(0.95), q_1: new NoulAnswer(0.85) },
});

/** Both barely past the fence: least decisive |2*0.55 - 1| = 0.10. */
const UNSURE = new SystemOneAnswers({
  nouls: { q_0: new NoulAnswer(0.6), q_1: new NoulAnswer(0.55) },
});

const VerdictsSchema = z.object({
  verdicts: z.array(z.object({ verdict: z.enum(["yes", "no"]) })),
});

const LLM_VERDICTS = { verdicts: [{ verdict: "yes" }, { verdict: "no" }] };

/**
 * The smallest chain metric: split `actualOutput` into sentences, judge
 * each (Jev per sentence under `hybrid`, the LLM otherwise), and score the
 * fraction judged relevant. Its whole-chain form is TOY_QUESTIONS.
 */
class ToyMetric extends BaseMetric {
  path?: "legacy" | "system_one";
  private readonly withSpec: boolean;

  constructor(
    options: {
      model?: DeepEvalBaseLLM;
      systemOneModel?: DeepEvalBaseSystemOneModel;
      evalMode?: EvalModeName;
      strictMode?: boolean;
      includeReason?: boolean;
      spec?: boolean;
    } = {},
  ) {
    super(options.strictMode ? 1 : 0.5, {
      strictMode: options.strictMode,
      includeReason: options.includeReason ?? true,
      showIndicator: false,
    });
    this.requiredParams = [
      SingleTurnParams.INPUT,
      SingleTurnParams.ACTUAL_OUTPUT,
    ];
    this.withSpec = options.spec ?? true;
    initializeMetricModels(this, {
      model: options.model ?? new ScriptedLLM(LLM_VERDICTS),
      systemOneModel: options.systemOneModel,
      evalMode: options.evalMode,
    });
  }

  systemOneEvalSpec(): SystemOneEvalSpec | undefined {
    if (!this.withSpec) return undefined;
    return { evaluationParams: this.requiredParams, questions: TOY_QUESTIONS };
  }

  async measure(testCase: LLMTestCase): Promise<number> {
    await this.startProgress();
    try {
      checkSingleTurnParams(testCase, this.requiredParams, this);
      this.evaluationCost = 0;
      if (await runSystemOneEval(this, testCase)) {
        this.path = "system_one";
        return this.score as number;
      }
      this.path = "legacy";
      const sentences = splitSentences(testCase.actualOutput);
      const verdicts = await generateQagVerdicts(this, {
        systemOne: {
          instructions: "Is `sentence` relevant to `input`?",
          items: sentences,
          itemKey: "sentence",
          state: { input: testCase.input },
        },
        llm: async () =>
          (await generateWithSchema(this, "judge", VerdictsSchema)).verdicts,
      });
      const passed = verdicts.filter((v) => v.verdict === "yes").length;
      this.score = verdicts.length === 0 ? 1 : passed / verdicts.length;
      this.reason = this.includeReason ? "LLM reason" : undefined;
      this.success = this.isSuccessful();
      return this.score;
    } finally {
      this.stopProgress();
    }
  }

  get name(): string {
    return "Toy";
  }
}

describe("eval mode resolution", () => {
  it("defaults to llm", () => {
    expect(resolveEvalMode()).toBe(EvalMode.LLM);
  });

  it("is not turned on by the experimental feature channel", () => {
    process.env.DEEPEVAL_MODE = "experimental";
    expect(resolveEvalMode()).toBe(EvalMode.LLM);
  });

  it("reads the setting independently of the feature channel", () => {
    process.env.DEEPEVAL_MODE = "experimental";
    process.env.DEEPEVAL_EVAL_MODE = "llm";
    expect(resolveEvalMode()).toBe(EvalMode.LLM);
    process.env.DEEPEVAL_MODE = "stable";
    process.env.DEEPEVAL_EVAL_MODE = "system_one";
    expect(resolveEvalMode()).toBe(EvalMode.SYSTEM_ONE);
    process.env.DEEPEVAL_EVAL_MODE = "hybrid";
    expect(resolveEvalMode()).toBe(EvalMode.HYBRID);
  });

  it("lets the metric option beat the setting", () => {
    process.env.DEEPEVAL_EVAL_MODE = "system_one";
    expect(resolveEvalMode("llm")).toBe(EvalMode.LLM);
    expect(resolveEvalMode("SYSTEM_ONE")).toBe(EvalMode.SYSTEM_ONE);
    expect(() => resolveEvalMode("system-one")).toThrow();
    expect(() => resolveEvalMode("jev")).toThrow();
  });

  it("treats an unrecognised setting as unset", () => {
    process.env.DEEPEVAL_EVAL_MODE = "banana";
    expect(resolveEvalMode()).toBe(EvalMode.LLM);
  });

  it("rejects an unrecognised option", () => {
    expect(() => resolveEvalMode("banana")).toThrow(/banana/);
  });

  it("builds no System One model under llm", () => {
    const metric = new ToyMetric({
      evalMode: "llm",
      systemOneModel: new FakeSystemOneModel(),
    });
    expect(metric.systemOneModel).toBeUndefined();
    expect(systemOneActive(metric, {})).toBe(false);
  });

  it("fails at construction when hybrid has no TypeSafe key", () => {
    expect(() => new ToyMetric({ evalMode: "hybrid" })).toThrow(
      /TYPESAFE_API_KEY/,
    );
  });

  it("derives a symmetric Noul confidence", () => {
    expect(new NoulAnswer(0.95).confidence).toBeCloseTo(0.9);
    expect(new NoulAnswer(0.05).confidence).toBeCloseTo(0.9);
    expect(new NoulAnswer(0.5).confidence).toBeCloseTo(0);
    expect(CONFIDENT.minConfidence()).toBeCloseTo(0.7);
  });
});

describe("llm mode", () => {
  it("runs the legacy chain only", async () => {
    const llm = new ScriptedLLM(LLM_VERDICTS);
    const metric = new ToyMetric({ model: llm, evalMode: "llm" });
    await metric.measure(TEST_CASE);
    expect(metric.path).toBe("legacy");
    expect(metric.score).toBe(0.5);
    expect(metric.reason).toBe("LLM reason");
    expect(metric.confidence).toBeUndefined();
    expect(metric.systemOneFallbackReason).toBeUndefined();
    expect(llm.prompts).toHaveLength(1);
  });
});

describe("hybrid mode", () => {
  it("asks Jev for the verdicts and keeps the LLM for the rest", async () => {
    const jev = new FakeSystemOneModel({ answerFn: noulAnswers(0.9) });
    const metric = new ToyMetric({
      model: new ExplodingLLM(),
      systemOneModel: jev,
      evalMode: "hybrid",
    });
    await metric.measure(TEST_CASE);
    expect(metric.path).toBe("legacy");
    expect(metric.score).toBe(1);
    expect(jev.calls).toHaveLength(1);
    expect(Object.keys(jev.calls[0].questions)).toEqual([
      "sentence_0",
      "sentence_1",
    ]);
    expect(jev.calls[0].state).toEqual({
      input: TEST_CASE.input,
      sentences: ["Paris is the capital of France.", "It has great food."],
    });
    expect(metric.confidence).toBeCloseTo(0.8);
    expect(metric.reason).toBe("LLM reason");
  });

  it("falls back to the LLM when a Jev call fails", async () => {
    const outage = Object.assign(new Error("jev is down"), {
      name: "APIConnectionError",
    });
    const llm = new ScriptedLLM(LLM_VERDICTS);
    const metric = new ToyMetric({
      model: llm,
      systemOneModel: new ExplodingSystemOneModel(outage),
      evalMode: "hybrid",
    });
    await metric.measure(TEST_CASE);
    expect(metric.score).toBe(0.5);
    expect(llm.prompts).toHaveLength(1);
    expect(metric.systemOneFallbackReason).toContain(
      "APIConnectionError: jev is down",
    );
  });

  it("falls back on a context overflow", async () => {
    const metric = new ToyMetric({
      systemOneModel: new ExplodingSystemOneModel(
        new SystemOneContextLimitError("too big", {
          estimatedTokens: 99_999,
          limitTokens: 32_000,
        }),
      ),
      evalMode: "hybrid",
    });
    await metric.measure(TEST_CASE);
    expect(metric.score).toBe(0.5);
    expect(metric.systemOneFallbackReason).toContain(
      "context limit (99999 est. tokens > 32000)",
    );
  });

  it("keeps the rest of the measure on the LLM after an auth failure", async () => {
    const jev = new ExplodingSystemOneModel(
      Object.assign(new Error("invalid key"), { status: 401 }),
    );
    const metric = new ToyMetric({ systemOneModel: jev, evalMode: "hybrid" });
    await metric.measure(TEST_CASE);
    expect(metric._systemOneDisabled).toBe(true);
    await metric.measure(TEST_CASE); // re-enabled for the next measure
    expect(jev.calls).toHaveLength(2);
  });

  it("never swallows programming errors", async () => {
    const metric = new ToyMetric({
      systemOneModel: new ExplodingSystemOneModel(new RangeError("sentence_7")),
      evalMode: "hybrid",
    });
    await expect(metric.measure(TEST_CASE)).rejects.toThrow(RangeError);
  });
});

describe("system_one mode", () => {
  it("decides the whole measure in one request with no LLM call", async () => {
    const jev = new FakeSystemOneModel({ answers: CONFIDENT });
    const metric = new ToyMetric({
      model: new ExplodingLLM(),
      systemOneModel: jev,
      evalMode: "system_one",
    });
    const score = await metric.measure(TEST_CASE);
    expect(metric.path).toBe("system_one");
    expect(score).toBeCloseTo((2 * 0.95 + 0.85) / 3);
    expect(metric.success).toBe(true);
    expect(metric.confidence).toBeCloseTo(0.7);
    expect(metric.systemOneFallbackReason).toBeUndefined();
    expect(metric.evaluationModel).toBe("fake-jev");
    expect(metric.scoreBreakdown).toHaveLength(2);
    expect(jev.calls).toHaveLength(1);
    expect(Object.keys(jev.calls[0].state.test_case).sort()).toEqual([
      "actual_output",
      "input",
    ]);
    expect(Object.keys(jev.calls[0].questions)).toEqual(["q_0", "q_1"]);
  });

  it("builds a deterministic reason", async () => {
    const metric = new ToyMetric({
      systemOneModel: new FakeSystemOneModel({ answers: CONFIDENT }),
      evalMode: "system_one",
    });
    await metric.measure(TEST_CASE);
    const reason = metric.reason as string;
    expect(
      reason.startsWith("Decided by fake-jev, minimum confidence 0.70."),
    ).toBe(true);
    expect(reason).toContain(
      "1. actual_output answers input. -> clearly holds (P(yes)=0.95, confidence=0.90, weight=2)",
    );
    expect(
      reason.endsWith("Score: 0.92 (weighted mean of 2 applicable questions)."),
    ).toBe(true);
  });

  it("honours includeReason: false", async () => {
    const metric = new ToyMetric({
      systemOneModel: new FakeSystemOneModel({ answers: CONFIDENT }),
      evalMode: "system_one",
      includeReason: false,
    });
    await metric.measure(TEST_CASE);
    expect(metric.path).toBe("system_one");
    expect(metric.reason).toBeUndefined();
  });

  it("applies strict mode", async () => {
    const metric = new ToyMetric({
      systemOneModel: new FakeSystemOneModel({ answers: UNSURE }),
      evalMode: "system_one",
      strictMode: true,
    });
    await metric.measure(TEST_CASE);
    expect(metric.threshold).toBe(1);
    expect(metric.score).toBe(1);
    expect((metric.scoreBreakdown as any[]).every((o) => o.passed)).toBe(true);
    expect(metric.reason).toContain("strict=pass");
  });

  it("builds no LLM, whatever model was passed", () => {
    const metric = new ToyMetric({
      model: new ExplodingLLM(),
      systemOneModel: new FakeSystemOneModel({ answers: CONFIDENT }),
      evalMode: "system_one",
    });
    expect(metric.model).toBeUndefined();
    expect(metric.evaluationModel).toBe("fake-jev");
  });

  it("keeps a low-confidence result", async () => {
    const metric = new ToyMetric({
      systemOneModel: new FakeSystemOneModel({ answers: UNSURE }),
      evalMode: "system_one",
    });
    await metric.measure(TEST_CASE);
    expect(metric.path).toBe("system_one");
    expect(metric.confidence).toBeCloseTo(0.1);
    expect(metric.systemOneFallbackReason).toBeUndefined();
  });

  it("surfaces Jev errors unchanged", async () => {
    const outage = Object.assign(new Error("jev is down"), {
      name: "APIConnectionError",
    });
    const metric = new ToyMetric({
      systemOneModel: new ExplodingSystemOneModel(outage),
      evalMode: "system_one",
    });
    await expect(metric.measure(TEST_CASE)).rejects.toBe(outage);
  });

  it("says to switch back to llm on a context overflow", async () => {
    const metric = new ToyMetric({
      systemOneModel: new ExplodingSystemOneModel(
        new SystemOneContextLimitError("too big", {
          estimatedTokens: 99_999,
          limitTokens: 32_000,
        }),
      ),
      evalMode: "system_one",
    });
    await expect(metric.measure(TEST_CASE)).rejects.toThrow(/evalMode: "llm"/);
  });

  it("checks the context budget before sending the request", async () => {
    const jev = new FakeSystemOneModel({ answers: CONFIDENT });
    const metric = new ToyMetric({
      systemOneModel: jev,
      evalMode: "system_one",
    });
    const huge = new LLMTestCase({
      input: "q",
      actualOutput: "x".repeat(200_000),
    });
    await expect(metric.measure(huge)).rejects.toThrow(/set-eval-mode llm/);
    expect(jev.calls).toEqual([]);
  });

  it("refuses a test case it has no whole-chain form for", async () => {
    const metric = new ToyMetric({
      systemOneModel: new FakeSystemOneModel({ answers: CONFIDENT }),
      evalMode: "system_one",
      spec: false,
    });
    await expect(metric.measure(TEST_CASE)).rejects.toThrow(
      /cannot judge this test case/,
    );
  });

  it("resets per-measure state", async () => {
    const jev = new FakeSystemOneModel({ answers: [UNSURE, CONFIDENT] });
    const metric = new ToyMetric({
      systemOneModel: jev,
      evalMode: "system_one",
    });
    await metric.measure(TEST_CASE);
    expect(metric.confidence).toBeCloseTo(0.1);
    await metric.measure(TEST_CASE);
    expect(metric.confidence).toBeCloseTo(0.7);
  });
});

describe("built-in metric: AnswerRelevancy", () => {
  const WHOLE_CHAIN = new SystemOneAnswers({
    nouls: { q_0: new NoulAnswer(0.9), q_1: new NoulAnswer(0.8) },
    scores: {
      q_2: new ScoreAnswer(2.7, { 0: 0, 1: 0.1, 2: 0.1, 3: 0.8 }, 0.73),
    },
  });

  it("runs the whole chain on Jev under system_one", async () => {
    const jev = new FakeSystemOneModel({ answers: WHOLE_CHAIN });
    const metric = new AnswerRelevancyMetric({
      model: new ExplodingLLM(),
      systemOneModel: jev,
      evalMode: "system_one",
      showIndicator: false,
    });
    const score = await metric.measure(TEST_CASE);
    expect(score).toBeCloseTo((1.8 + 0.8 + 0.9) / 4);
    expect(metric.reason?.startsWith("Decided by fake-jev")).toBe(true);
    expect(metric.reason).toContain(
      "Every statement in `actual_output` is relevant",
    );
    expect(metric.confidence).toBeCloseTo(0.6);
    expect(metric.evaluationModel).toBe("fake-jev");
    expect(jev.calls).toHaveLength(1);
    expect(Object.keys(jev.calls[0].state.test_case).sort()).toEqual([
      "actual_output",
      "input",
    ]);
    expect(Object.keys(jev.calls[0].questions)).toHaveLength(3);
    expect(metric.verboseLogs).toContain("System One");
  });

  it("surfaces Jev errors under system_one", async () => {
    const outage = Object.assign(new Error("down"), {
      name: "APIConnectionError",
    });
    const jev = new ExplodingSystemOneModel(outage);
    const metric = new AnswerRelevancyMetric({
      systemOneModel: jev,
      evalMode: "system_one",
      showIndicator: false,
    });
    await expect(metric.measure(TEST_CASE)).rejects.toThrow("down");
    expect(jev.calls).toHaveLength(1);
  });

  it("asks Jev per statement under hybrid", async () => {
    const jev = new FakeSystemOneModel({ answerFn: noulAnswers(0.9) });
    const llm = new ScriptedLLM((prompt: string) =>
      prompt.includes("statements")
        ? { statements: ["Paris is the capital of France."], reason: "fine" }
        : { reason: "fine" },
    );
    const metric = new AnswerRelevancyMetric({
      model: llm,
      systemOneModel: jev,
      evalMode: "hybrid",
      showIndicator: false,
    });
    expect(await metric.measure(TEST_CASE)).toBe(1);
    expect(metric.reason).toBe("fine");
    expect(jev.calls).toHaveLength(1);
    expect(Object.keys(jev.calls[0].questions)).toEqual(["statement_0"]);
    expect(metric.confidence).toBeCloseTo(0.8);
    expect(metric.systemOneFallbackReason).toBeUndefined();
  });

  it("never touches Jev under llm", async () => {
    const llm = new ScriptedLLM({
      statements: ["Paris is the capital of France."],
      verdicts: [{ verdict: "yes" }],
      reason: "fine",
    });
    const metric = new AnswerRelevancyMetric({
      model: llm,
      evalMode: "llm",
      showIndicator: false,
    });
    expect(metric.systemOneModel).toBeUndefined();
    expect(await metric.measure(TEST_CASE)).toBe(1);
    expect(metric.confidence).toBeUndefined();
  });

  it("rejects a multimodal test case under system_one", async () => {
    const jev = new FakeSystemOneModel({ answerFn: noulAnswers(0.9) });
    const metric = new AnswerRelevancyMetric({
      systemOneModel: jev,
      evalMode: "system_one",
      showIndicator: false,
    });
    await expect(
      metric.measure(
        new LLMTestCase({
          input: "what is this?",
          actualOutput: "a cat",
          multimodal: true,
        }),
      ),
    ).rejects.toThrow(/text only/);
    expect(jev.calls).toEqual([]);
  });
});

describe("built-in metric: DAG", () => {
  function gate() {
    const node = new BinaryJudgementNode({
      criteria: "Is the answer correct?",
      evaluationParams: [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
      ],
    });
    node.addVerdict(true, { score: 10 });
    node.addVerdict(false, { score: 0 });
    return node;
  }

  it("runs as hybrid under system_one: Jev judges, the LLM stays built", async () => {
    const jev = new FakeSystemOneModel({ answerFn: noulAnswers(0.9) });
    const llm = new ScriptedLLM({ verdict: false, reason: "llm" });
    const metric = new DAGMetric({
      name: "Correctness",
      dag: new DeepAcyclicGraph({ rootNodes: [gate()] }),
      model: llm,
      systemOneModel: jev,
      evalMode: "system_one",
      includeReason: false,
      showIndicator: false,
    });
    expect(effectiveEvalMode(metric)).toBe(EvalMode.HYBRID);
    expect(metric.model).toBe(llm);
    expect(await metric.measure(TEST_CASE)).toBe(1);
    expect(jev.calls).toHaveLength(1);
    expect(jev.calls[0].state).toEqual({
      text: expect.stringContaining("Paris is the capital of France."),
    });
    expect(llm.prompts).toHaveLength(0);
  });

  it("falls back to the LLM when the judgement's Jev call fails", async () => {
    const llm = new ScriptedLLM({ verdict: false, reason: "llm" });
    const metric = new DAGMetric({
      name: "Correctness",
      dag: new DeepAcyclicGraph({ rootNodes: [gate()] }),
      model: llm,
      systemOneModel: new ExplodingSystemOneModel(
        new DeepEvalError("rate limited"),
      ),
      evalMode: "hybrid",
      includeReason: false,
      showIndicator: false,
    });
    expect(await metric.measure(TEST_CASE)).toBe(0);
    expect(llm.prompts).toHaveLength(1);
    expect(metric.systemOneFallbackReason).toContain("rate limited");
  });
});

describe("built-in metric: GEval", () => {
  it("stays on the LLM in every eval mode", async () => {
    process.env.DEEPEVAL_EVAL_MODE = "system_one";
    const llm = new ScriptedLLM({ score: 7, reason: "llm says so" });
    const metric = new GEval({
      name: "Helpfulness",
      evaluationParams: [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
      ],
      evaluationSteps: ["Check it answers.", "Check it is polite."],
      model: llm,
      showIndicator: false,
    });
    expect(metric.systemOneModel).toBeUndefined();
    expect(await metric.measure(TEST_CASE)).toBeCloseTo(0.7);
    expect(metric.reason).toBe("llm says so");
    expect(metric.confidence).toBeUndefined();
  });
});
