// Batched System One in `evaluate()`: every `system_one` metric on a test case
// shares one Jev request (a port of Python's
// tests/test_metrics/test_system_one_batch.py).

import * as fs from "fs";
import * as path from "path";
import {
  AnswerRelevancyMetric,
  FaithfulnessMetric,
  MultiTurnMCPUseMetric,
  RoleAdherenceMetric,
  ToolCorrectnessMetric,
  TurnFaithfulnessMetric,
  TurnRelevancyMetric,
} from "@/metrics";
import { evaluate } from "@/evaluate/evaluate";
import { measureSystemOneBatch } from "@/metrics/system-one/batch";
import * as limits from "@/models/system-one/limits";
import { SystemOneContextLimitError } from "@/models/system-one/limits";
import {
  ConversationalTestCase,
  LLMTestCase,
  ToolCall,
  Turn,
} from "@/test-case";
import {
  ExplodingLLM,
  ExplodingSystemOneModel,
  FakeSystemOneModel,
  answerEverything,
} from "./system-one-fakes";

beforeEach(() => {
  delete process.env.DEEPEVAL_EVAL_MODE;
  delete process.env.DEEPEVAL_MODE;
});

afterEach(() => jest.restoreAllMocks());

const jev = (name = "fake-jev", cost: number | null = 0) =>
  new FakeSystemOneModel({ answerFn: answerEverything(), name, cost });

type MetricClass = new (options: any) => any;

const jevJudged = <T extends MetricClass>(
  cls: T,
  systemOneModel: FakeSystemOneModel,
  options: Record<string, unknown> = {},
): InstanceType<T> =>
  new cls({
    model: new ExplodingLLM(),
    systemOneModel,
    evalMode: "system_one",
    showIndicator: false,
    ...options,
  });

const makeCase = (overrides: Record<string, unknown> = {}) =>
  new LLMTestCase({
    input: "Where is the Eiffel Tower?",
    actualOutput: "The Eiffel Tower is in Paris.",
    retrievalContext: ["The Eiffel Tower stands in Paris, France."],
    ...overrides,
  });

const makeConversation = (overrides: Record<string, unknown> = {}) =>
  new ConversationalTestCase({
    chatbotRole: "A concise travel guide.",
    turns: [
      new Turn({ role: "user", content: "Where is the Eiffel Tower?" }),
      new Turn({
        role: "assistant",
        content: "It is in Paris.",
        retrievalContext: ["The Eiffel Tower stands in Paris."],
      }),
    ],
    ...overrides,
  });

const run = (
  metrics: any[],
  testCase: LLMTestCase | ConversationalTestCase,
  options: { ignoreErrors?: boolean; skipOnMissingParams?: boolean } = {},
) =>
  measureSystemOneBatch(metrics, testCase, {
    ignoreErrors: options.ignoreErrors ?? false,
    skipOnMissingParams: options.skipOnMissingParams ?? false,
  });

describe("single-turn", () => {
  it("shares one request across metrics", async () => {
    const model = jev();
    const metrics = [
      jevJudged(AnswerRelevancyMetric, model),
      jevJudged(FaithfulnessMetric, model),
    ];

    expect(await run(metrics, makeCase())).toEqual(metrics);
    expect(model.calls).toHaveLength(1);
    const { state, questions } = model.calls[0];
    expect(Object.keys(state.test_case).sort()).toEqual([
      "actual_output",
      "input",
      "retrieval_context",
    ]);
    expect(
      new Set(Object.keys(questions).map((key) => key.split(".")[0])),
    ).toEqual(new Set(["m0", "m1"]));
  });

  it("matches separate measures", async () => {
    const model = jev();
    const batched = [
      jevJudged(AnswerRelevancyMetric, model, { threshold: 0.3 }),
      jevJudged(FaithfulnessMetric, model),
    ];
    await run(batched, makeCase());

    const alone = [
      jevJudged(AnswerRelevancyMetric, jev(), { threshold: 0.3 }),
      jevJudged(FaithfulnessMetric, jev()),
    ];
    for (const [i, metric] of alone.entries()) {
      await metric.measure(makeCase());
      expect(batched[i].score).toBe(metric.score);
      expect(batched[i].reason).toBe(metric.reason);
      expect(batched[i].success).toBe(metric.success);
      expect(batched[i].confidence).toBe(metric.confidence);
    }
  });

  it("leaves a single Jev metric to its own measure", async () => {
    const model = jev();
    expect(
      await run([jevJudged(AnswerRelevancyMetric, model)], makeCase()),
    ).toEqual([]);
    expect(model.calls).toHaveLength(0);
  });

  it("honours an evalMode override", async () => {
    const model = jev();
    const onLlm = jevJudged(AnswerRelevancyMetric, model, { evalMode: "llm" });
    const metrics = [
      jevJudged(AnswerRelevancyMetric, model),
      jevJudged(FaithfulnessMetric, model),
      onLlm,
    ];

    expect(await run(metrics, makeCase())).toEqual(metrics.slice(0, 2));
    expect(model.calls).toHaveLength(1);
  });

  it("gives each Jev model its own request", async () => {
    const first = jev("jev-a");
    const second = jev("jev-b");
    await run(
      [
        jevJudged(AnswerRelevancyMetric, first),
        jevJudged(FaithfulnessMetric, second),
      ],
      makeCase(),
    );
    expect(first.calls).toHaveLength(1);
    expect(second.calls).toHaveLength(1);
  });

  it("splits a request over the token budget", async () => {
    jest
      .spyOn(limits, "checkContextBudget")
      .mockImplementation((_state, questions) => {
        if (Object.keys(questions).length > 3) {
          throw new SystemOneContextLimitError("too big", {
            estimatedTokens: 2,
            limitTokens: 1,
          });
        }
      });
    const model = jev();
    const metrics = [
      jevJudged(AnswerRelevancyMetric, model),
      jevJudged(FaithfulnessMetric, model),
    ];

    await run(metrics, makeCase());

    expect(model.calls).toHaveLength(2);
    expect(metrics.every((m) => m.score !== undefined)).toBe(true);
  });

  it("errors only the metrics of a failed request", async () => {
    const down = new ExplodingSystemOneModel(new Error("jev is down"));
    const failed = [
      jevJudged(AnswerRelevancyMetric, down),
      jevJudged(FaithfulnessMetric, down),
    ];
    const judged = jevJudged(AnswerRelevancyMetric, jev("healthy-jev"));

    await run([...failed, judged], makeCase(), { ignoreErrors: true });

    expect(down.calls).toHaveLength(1);
    for (const metric of failed) expect(metric.error).toContain("jev is down");
    expect(judged.error).toBeUndefined();
    expect(judged.score).toBeDefined();
  });

  it("throws a failed request without ignoreErrors", async () => {
    const down = new ExplodingSystemOneModel(new Error("jev is down"));
    await expect(
      run(
        [
          jevJudged(AnswerRelevancyMetric, down),
          jevJudged(FaithfulnessMetric, down),
        ],
        makeCase(),
      ),
    ).rejects.toThrow("jev is down");
  });

  it("splits the cost by question share", async () => {
    const model = jev("fake-jev", 0.6);
    const metrics = [
      jevJudged(AnswerRelevancyMetric, model),
      jevJudged(FaithfulnessMetric, model),
    ];
    // A native model zeroes its running cost, so the shares have a total to add to.
    for (const metric of metrics) metric.usingNativeModel = true;

    await run(metrics, makeCase());

    const total = metrics.reduce((sum, m) => sum + (m.evaluationCost ?? 0), 0);
    expect(total).toBeCloseTo(0.6);
  });

  it("skips a metric missing a required param", async () => {
    const model = jev();
    const missingContext = jevJudged(FaithfulnessMetric, model);
    const metrics = [
      jevJudged(AnswerRelevancyMetric, model),
      missingContext,
      jevJudged(AnswerRelevancyMetric, model),
    ];

    const handled = await run(
      metrics,
      makeCase({ retrievalContext: undefined }),
      { skipOnMissingParams: true },
    );

    expect(handled).toEqual(metrics);
    expect(missingContext.skipped).toBe(true);
    expect(model.calls).toHaveLength(1);
  });

  it("leaves a metric without a whole Jev request to its own measure", async () => {
    const model = jev();
    const toolCorrectness = jevJudged(ToolCorrectnessMetric, model);
    const metrics = [
      jevJudged(AnswerRelevancyMetric, model),
      jevJudged(FaithfulnessMetric, model),
      toolCorrectness,
    ];
    const testCase = makeCase({
      toolsCalled: [new ToolCall({ name: "search" })],
      expectedTools: [new ToolCall({ name: "search" })],
    });

    expect(await run(metrics, testCase)).toEqual(metrics.slice(0, 2));
    expect(toolCorrectness.error).toBeUndefined();
    expect(model.calls).toHaveLength(1);
  });
});

describe("conversational", () => {
  it("shares one request and merges each turn's fields", async () => {
    const model = jev();
    const metrics = [
      jevJudged(TurnRelevancyMetric, model),
      jevJudged(TurnFaithfulnessMetric, model),
    ];

    expect(await run(metrics, makeConversation())).toEqual(metrics);
    expect(model.calls).toHaveLength(1);
    const { turns } = model.calls[0].state;
    expect(turns).toHaveLength(2);
    expect(turns[1].content).toBe("It is in Paris.");
    expect(
      turns[1].retrieval_context ?? turns[1].retrievalContext,
    ).toBeTruthy();

    for (const [i, cls] of [
      TurnRelevancyMetric,
      TurnFaithfulnessMetric,
    ].entries()) {
      const alone = jevJudged(cls, jev());
      await alone.measure(makeConversation());
      expect(metrics[i].score).toBe(alone.score);
      expect(metrics[i].reason).toBe(alone.reason);
    }
  });

  it("skips metrics whose required conversation fields are missing", async () => {
    const model = jev();
    const roleAdherence = jevJudged(RoleAdherenceMetric, model);
    const mcpUse = jevJudged(MultiTurnMCPUseMetric, model);
    const metrics = [
      jevJudged(TurnRelevancyMetric, model),
      roleAdherence,
      mcpUse,
      jevJudged(TurnFaithfulnessMetric, model),
    ];

    await run(metrics, makeConversation({ chatbotRole: undefined }), {
      skipOnMissingParams: true,
    });

    expect(roleAdherence.skipped).toBe(true);
    expect(mcpUse.skipped).toBe(true);
    expect(model.calls).toHaveLength(1);
  });
});

describe("evaluate()", () => {
  const options = {
    displayConfig: { showIndicator: false, printResults: false },
    cacheConfig: { useCache: false, writeCache: false },
  };

  it.each([
    [
      "single-turn",
      [makeCase(), makeCase({ input: "What is in Paris?" })],
      [AnswerRelevancyMetric, FaithfulnessMetric],
    ],
    [
      "conversational",
      [makeConversation(), makeConversation({ chatbotRole: "A curt guide." })],
      [TurnRelevancyMetric, TurnFaithfulnessMetric],
    ],
  ] as const)(
    "%s: sends one Jev request per test case",
    async (_kind, testCases, classes) => {
      const model = jev();
      const { testResults } = await evaluate(
        [...testCases],
        classes.map((cls) => jevJudged(cls as MetricClass, model)),
        options,
      );

      expect(model.calls).toHaveLength(testCases.length);
      for (const result of testResults) {
        expect(result.metricsData?.map((m) => m.score !== undefined)).toEqual([
          true,
          true,
        ]);
      }
    },
  );
});

it("every whole-Jev metric sets up through prepareMeasure", () => {
  const root = path.join(__dirname, "../../src/metrics");
  const walk = (dir: string): string[] =>
    fs
      .readdirSync(dir, { withFileTypes: true })
      .flatMap((entry) =>
        entry.isDirectory()
          ? walk(path.join(dir, entry.name))
          : [path.join(dir, entry.name)],
      );
  const stale = walk(root)
    .filter((file) => file.endsWith(".ts"))
    .filter((file) => {
      const src = fs.readFileSync(file, "utf8");
      return (
        /\n  systemOneEvalSpec\(/.test(src) &&
        /check(SingleTurnParams|ConversationalTestCaseParams)\(testCase/.test(
          src,
        )
      );
    })
    .map((file) => path.relative(root, file));
  expect(stale).toEqual([]);
});
