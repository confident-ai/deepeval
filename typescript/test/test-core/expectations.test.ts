import { observe, updateCurrentTrace } from "@/tracing";
import { runMetrics } from "@/evaluate/test-run/run-metrics";
import { z } from "zod";
import {
  Expectations,
  Golden,
  ConversationalGolden,
  EvaluationDataset,
} from "@/dataset";
import { LLMTestCase, ConversationalTestCase, Turn } from "@/test-case";
import { DeepEvalBaseLLM, type GenerationResult } from "@/models";
import { evaluate } from "@/evaluate";
import { evaluateCase } from "@/evaluate/test-run/run-metrics";
import { buildTestCaseEntry } from "@/evaluate/confident";
import { ExpectationKind } from "@/evaluate/types";
import { Api } from "@/confident/api";
import { withExpectations } from "@/evaluate/expectations";
import { testCaseCacheKey } from "@/evaluate/test-run/cache";
import * as utils from "@/metrics/utils";
import { FakeSystemOneModel } from "./system-one-fakes";
import { ChoiceAnswer, SystemOneAnswers } from "@/models/system-one";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

class Judge extends DeepEvalBaseLLM {
  prompts: string[] = [];
  reasonPrompts: string[] = [];
  constructor(
    private status = "pass",
    private invalid = false,
  ) {
    super("expectation-judge");
  }
  getModelName() {
    return "expectation-judge";
  }
  async generate<T>(
    prompt: string,
    _schema?: z.ZodType<T>,
  ): Promise<GenerationResult<T>> {
    if (!prompt.includes("Observed case:")) {
      this.reasonPrompts.push(prompt);
      return { cost: 0, output: { reason: "Judged reason" } as T };
    }
    this.prompts.push(prompt);
    const requirements = JSON.parse(
      prompt.split("Requirements: ")[1].split("\nObserved case:")[0],
    );
    return {
      cost: 0,
      output: {
        verdicts: requirements.map((r: { id: string }) => ({
          id: this.invalid ? "invalid" : r.id,
          status: this.status,
          reason: "Judged",
          evidence: "Observed",
        })),
      } as T,
    };
  }
}
const options = {
  displayConfig: { printResults: false, showIndicator: false },
  cacheConfig: { useCache: false, writeCache: false },
};
beforeEach(() => {
  delete process.env.DEEPEVAL_EVAL_MODE;
  jest.spyOn(console, "warn").mockImplementation(() => {});
});
afterEach(() => jest.restoreAllMocks());
const single = (expectations?: Expectations) =>
  new LLMTestCase({ input: "Hi", actualOutput: "Hello", expectations });

test("validates conditions and permits either list alone", () => {
  expect(new Expectations({ must: ["  Greet  "] }).must).toEqual(["Greet"]);
  expect(new Expectations({ mustNot: ["Insult"] }).hasConditions).toBe(true);
  expect(new Expectations().hasConditions).toBe(false);
  expect(() => new Expectations({ must: [""] })).toThrow("non-empty");
  expect(() => new Expectations({ evalMode: "hybrid" as any })).toThrow(
    "evalMode",
  );
  expect(() => new Expectations({ extra: true } as any)).toThrow("Unknown");
});

test.each([false, true])(
  "evaluates expectations without metrics (conversation=%s)",
  async (conversation) => {
    const model = new Judge();
    const expectations = new Expectations({
      must: ["Greet"],
      mustNot: ["Insult"],
      model,
    });
    const tc = conversation
      ? new ConversationalTestCase({
          turns: [new Turn({ role: "assistant", content: "Hello" })],
          expectations,
        })
      : single(expectations);
    const result = await evaluate([tc], undefined, options);
    expect(result.testResults[0].success).toBe(true);
    expect(result.testResults[0].metricsData?.[0].name).toBe("Expectations");
    expect(result.confidentLink).toBeNull();
    expect(model.prompts).toHaveLength(1);
  },
);

test("missing coverage leads with the count", async () => {
  await expect(evaluate([single(), single()], [], options)).rejects.toThrow(
    "2 test cases are missing expectations.",
  );
});

test.each(["fail", "unable_to_evaluate"])("handles %s", async (status) => {
  const result = await evaluate(
    [single(new Expectations({ must: ["Greet"], model: new Judge(status) }))],
    [],
    { ...options, errorConfig: { ignoreErrors: true } },
  );
  expect(result.testResults[0].success).toBe(false);
  if (status === "unable_to_evaluate")
    expect(result.testResults[0].metricsData?.[0].error).toContain(
      "Unable to evaluate",
    );
});

test("rejects invalid verdict IDs", async () => {
  await expect(
    evaluate(
      [
        single(
          new Expectations({ must: ["Greet"], model: new Judge("pass", true) }),
        ),
      ],
      [],
      options,
    ),
  ).rejects.toThrow("verdict IDs");
});

test("assertion runner allows expectations alone", async () => {
  const evaluated = await evaluateCase(
    single(new Expectations({ mustNot: ["Insult"], model: new Judge() })),
    [],
  );
  expect(evaluated.metricsData[0].success).toBe(true);
});

test("global mode overrides local mode and hybrid maps to llm", () => {
  process.env.DEEPEVAL_EVAL_MODE = "hybrid";
  const tc = single(
    new Expectations({
      must: ["Greet"],
      model: new Judge(),
      evalMode: "system_one",
    }),
  );
  expect(withExpectations([], tc)[0].evalMode).toBe("llm");
});

test.each(["pass", "unable_to_evaluate"])(
  "system one %s uses shared judge utilities without LLM",
  async (answer) => {
    const jev = new FakeSystemOneModel({
      answerFn: (questions) =>
        new SystemOneAnswers({
          choices: Object.fromEntries(
            Object.keys(questions).map((key) => [
              key,
              new ChoiceAnswer(
                answer,
                {
                  pass: answer === "pass" ? 1 : 0,
                  fail: 0,
                  unable_to_evaluate: answer === "pass" ? 0 : 1,
                },
                1,
              ),
            ]),
          ),
        }),
    });
    const initialize = utils.initializeMetricModels;
    jest
      .spyOn(utils, "initializeMetricModels")
      .mockImplementation((metric, options) =>
        initialize(metric, { ...options, systemOneModel: jev }),
      );
    process.env.DEEPEVAL_EVAL_MODE = "system_one";
    const model = new Judge();
    const result = await evaluate(
      [single(new Expectations({ must: ["Greet"], model, evalMode: "llm" }))],
      [],
      { ...options, errorConfig: { ignoreErrors: true } },
    );
    expect(result.testResults[0].success).toBe(answer === "pass");
    expect(model.prompts).toHaveLength(0);
    expect(jev.calls).toHaveLength(1);
  },
);

test.each([false, true])(
  "uploads expectationsData instead of the Expectations metric (conversation=%s)",
  async (conversation) => {
    const send = jest
      .spyOn(Api.prototype, "sendRequest")
      .mockResolvedValue({ link: "link", id: "run-id" });
    const expectations = new Expectations({
      must: ["Greet"],
      mustNot: ["Insult"],
      model: new Judge(),
    });
    const tc = conversation
      ? new ConversationalTestCase({
          turns: [new Turn({ role: "assistant", content: "Hello" })],
          expectations,
        })
      : single(expectations);
    process.env.CONFIDENT_API_KEY = "fake-key";
    try {
      const result = await evaluate([tc], undefined, options);
      // Locally the check still reports like any metric.
      expect(result.testResults[0].metricsData!.map((m) => m.name)).toContain(
        "Expectations",
      );
    } finally {
      delete process.env.CONFIDENT_API_KEY;
    }

    const payload = send.mock.calls[0][2] as Record<string, any>;
    const [uploaded] = [
      ...payload.testCases,
      ...payload.conversationalTestCases,
    ];
    expect(uploaded).not.toHaveProperty("expectations");
    expect(uploaded.metricsData).toEqual([]);
    expect(payload.metricsScores).toEqual([]);
    expect(uploaded.expectationsData).toMatchObject({
      success: true,
      score: 1,
      reason: "Judged reason",
      evaluationModel: "expectation-judge",
      verdicts: [
        { kind: "MUST", condition: "Greet", status: "pass" },
        { kind: "MUST_NOT", condition: "Insult", status: "pass" },
      ],
    });
  },
);

test("an errored check uploads its verdicts and error", () => {
  const { entry } = buildTestCaseEntry(
    {
      testCase: single(new Expectations({ must: ["Greet"] })),
      metricsData: [
        {
          name: "Expectations",
          threshold: 1,
          success: false,
          strictMode: true,
          flaky: false,
          skipped: false,
          error: "Unable to evaluate expectations",
          evaluationCost: 0,
          expectationsData: {
            success: false,
            error: "Unable to evaluate expectations",
            evaluationCost: 0.01,
            verdicts: [
              {
                kind: ExpectationKind.MUST,
                condition: "Greet",
                status: "unable_to_evaluate",
                reason: "Judged",
                evidence: "",
              },
            ],
          },
        },
      ],
      runDuration: 0,
    },
    0,
  );
  expect(entry.metricsData).toEqual([]);
  expect(entry.success).toBe(false);
  // Cost follows the metric data, which a cache hit zeroes.
  expect(entry.expectationsData).toMatchObject({
    error: "Unable to evaluate expectations",
    evaluationCost: 0,
    verdicts: [{ status: "unable_to_evaluate" }],
  });
});

test.each([false, true])(
  "push, queue and update keep judge config local (conversation=%s)",
  async (conversation) => {
    const send = jest
      .spyOn(Api.prototype, "sendRequest")
      .mockResolvedValue({});
    jest.spyOn(console, "log").mockImplementation(() => {});
    const expectations = new Expectations({
      must: ["Cite a source"],
      mustNot: ["Mention competitors"],
      model: new Judge(),
      evalMode: "llm",
    });
    const golden = conversation
      ? new ConversationalGolden({ scenario: "Hi", expectations })
      : new Golden({ input: "Hi", expectations });
    const dataset = new EvaluationDataset({
      goldens:
        golden instanceof ConversationalGolden ? [golden] : [golden as Golden],
    });
    process.env.CONFIDENT_API_KEY = "fake-key";
    try {
      await dataset.push({ alias: "alias" });
      await dataset.queue({ alias: "alias", goldens: [golden] });
      golden.id = "golden-id";
      await dataset.updateGolden({ golden, alias: "alias" });
    } finally {
      delete process.env.CONFIDENT_API_KEY;
    }

    const key = conversation ? "conversationalGoldens" : "goldens";
    const bodies = send.mock.calls.map((call) => call[2] as any);
    for (const goldenBody of [bodies[0][key][0], bodies[1][key][0], bodies[2]]) {
      expect(goldenBody.expectations).toEqual({
        must: ["Cite a source"],
        mustNot: ["Mention competitors"],
      });
    }
    // The user's golden keeps its local judge configuration.
    expect(golden.expectations?.evalMode).toBe("llm");
    expect(golden.expectations?.model).toBeInstanceOf(Judge);
  },
);

test.each([false, true])(
  "pull reads expectations (conversation=%s)",
  async (conversation) => {
    const golden = {
      ...(conversation ? { scenario: "Hi" } : { input: "Hi" }),
      expectations: { must: ["Cite a source"], mustNot: ["Insult"] },
    };
    jest.spyOn(Api.prototype, "sendRequest").mockResolvedValue({
      id: "dataset",
      [conversation ? "conversationalGoldens" : "goldens"]: [golden],
    });
    process.env.CONFIDENT_API_KEY = "fake-key";
    const dataset = new EvaluationDataset();
    try {
      await dataset.pull({ alias: "alias" });
    } finally {
      delete process.env.CONFIDENT_API_KEY;
    }

    const [pulled] = dataset.goldens;
    expect(pulled.expectations?.must).toEqual(["Cite a source"]);
    expect(pulled.expectations?.mustNot).toEqual(["Insult"]);
  },
);

test("cache includes conditions and action evidence, serialization omits clients", () => {
  const tc = single(new Expectations({ must: ["Greet"], model: new Judge() }));
  const initial = testCaseCacheKey(tc);
  tc.expectations!.must[0] = "Refuse";
  expect(testCaseCacheKey(tc)).not.toBe(initial);
  const second = testCaseCacheKey(tc);
  tc._traceDict = { output: "changed" };
  expect(testCaseCacheKey(tc)).not.toBe(second);
  expect(JSON.parse(JSON.stringify(tc.expectations)).model).toBeNull();
});

test.each(["json", "jsonl", "csv"] as const)(
  "dataset %s round trip preserves expectations",
  async (fileType) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "expectations-"));
    try {
      for (const conversation of [false, true]) {
        const expectations = new Expectations({
          mustNot: ["Insult"],
          model: "gpt-4.1",
          evalMode: "llm",
        });
        const golden = conversation
          ? new ConversationalGolden({ scenario: "Greeting", expectations })
          : new Golden({ input: "Hi", expectations });
        const dataset = new EvaluationDataset({ goldens: [golden] as any });
        const filePath = await dataset.saveAs({ fileType, directory: dir });
        const loaded = new EvaluationDataset();
        if (fileType === "json") await loaded.addGoldensFromJSON({ filePath });
        if (fileType === "jsonl")
          await loaded.addGoldensFromJSONL({ filePath });
        if (fileType === "csv") await loaded.addGoldensFromCSV({ filePath });
        expect(loaded.goldens[0].expectations?.toJSON()).toEqual(
          expectations.toJSON(),
        );
      }
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  },
);

test("golden trace assertion evaluates expectations with observed evidence", async () => {
  const model = new Judge();
  const golden = new Golden({
    input: "Hi",
    expectations: new Expectations({ must: ["Greet"], model }),
  });
  const agent = observe({
    type: "agent",
    fn: async (input: string) => {
      updateCurrentTrace({ input, output: "Hello" });
      return "Hello";
    },
  });
  const outcome = await runMetrics(golden, [], { task: (g) => agent(g.input) });
  expect(outcome.pass).toBe(true);
  expect(model.prompts).toHaveLength(1);
  expect(model.prompts[0]).toContain('"trace"');
});

test("iterator evaluates golden expectations", async () => {
  const model = new Judge();
  const dataset = new EvaluationDataset({
    goldens: [
      new Golden({
        input: "Hi",
        expectations: new Expectations({ must: ["Greet"], model }),
      }),
    ],
  });
  const agent = observe({
    type: "agent",
    fn: async (input: string) => {
      updateCurrentTrace({ input, output: "Hello" });
      return "Hello";
    },
  });
  for await (const golden of dataset.evalsIterator({
    displayConfig: { showIndicator: false, printResults: false },
  })) {
    await agent((golden as Golden).input);
  }
  expect(model.prompts).toHaveLength(1);
});

test("different cases use their own judges", async () => {
  const passModel = new Judge();
  const failModel = new Judge("fail");
  const result = await evaluate(
    [
      single(new Expectations({ must: ["Greet"], model: passModel })),
      single(new Expectations({ mustNot: ["Insult"], model: failModel })),
    ],
    [],
    options,
  );
  expect(result.testResults.map((r) => r.success)).toEqual([true, false]);
  expect(passModel.prompts).toHaveLength(1);
  expect(failModel.prompts).toHaveLength(1);
});

test("measure exposes verdicts in condition order and writes a reason", async () => {
  const model = new Judge();
  const testCase = single(
    new Expectations({ must: ["Greet"], mustNot: ["Insult"], model }),
  );
  const [evaluator] = withExpectations([], testCase) as any[];
  await evaluator.measure(testCase);
  expect(
    evaluator.verdicts.map((v: { id: string; status: string }) => [
      v.id,
      v.status,
    ]),
  ).toEqual([
    ["must[0]", "pass"],
    ["mustNot[0]", "pass"],
  ]);
  expect(evaluator.reason).toBe("Judged reason");
  expect(model.reasonPrompts).toHaveLength(1);
  expect(model.reasonPrompts[0]).toContain("Score:\n1");
  expect(model.reasonPrompts[0]).toContain("Insult");
});

test("unable to evaluate keeps verdicts and skips the reason pass", async () => {
  const model = new Judge("unable_to_evaluate");
  const testCase = single(new Expectations({ must: ["Greet"], model }));
  const [evaluator] = withExpectations([], testCase) as any[];
  await expect(evaluator.measure(testCase)).rejects.toThrow(
    "Unable to evaluate expectations",
  );
  expect(evaluator.verdicts.map((v: { status: string }) => v.status)).toEqual(
    ["unable_to_evaluate"],
  );
  expect(evaluator.reason).toBeUndefined();
  expect(model.reasonPrompts).toHaveLength(0);
});
