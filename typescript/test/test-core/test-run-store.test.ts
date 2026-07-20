import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  persistCase,
  readPersistedCases,
  wrapUpTestRun,
} from "../../src/evaluate/assert-test/test-run-utils";
import { _resetWorkerCaseCount } from "../../src/evaluate/assert-test/test-run-utils";
import { assertTest } from "../../src/evaluate/assert-test";
import { LLMTestCase, ConversationalTestCase, Turn } from "../../src/test-case";
import { BaseMetric } from "../../src/metrics";
import { EvaluatedCase, MetricData } from "../../src/evaluate/types";
import { getIsRunningDeepEval, setIsRunningDeepEval } from "../../src/utils";
import { DEEPEVAL_RESULTS_DIR } from "../../src/constants";

class FakeMetric extends BaseMetric {
  private readonly ok: boolean;
  constructor(ok: boolean) {
    super(0.5);
    this.ok = ok;
  }
  get name(): string {
    return "FakeMetric";
  }
  isSuccessful(): boolean {
    return this.success ?? false;
  }
  measure(): number {
    this.score = this.ok ? 0.9 : 0.1;
    this.success = this.ok;
    return this.score;
  }
}

const metricData = (success: boolean): MetricData => ({
  name: "M",
  threshold: 0.5,
  success,
  score: success ? 0.9 : 0.1,
  strictMode: false,
  skipped: false,
});

const llmCase = (success = true): EvaluatedCase => ({
  testCase: new LLMTestCase({ input: "in", actualOutput: "out" }),
  metricsData: [metricData(success)],
  runDuration: 0.1,
});

describe("test-run-store", () => {
  let dir: string;
  const wasRunning = getIsRunningDeepEval();
  const savedApiKey = process.env.CONFIDENT_API_KEY;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-test-"));
    process.env[DEEPEVAL_RESULTS_DIR] = dir;
    // Keep unit tests hermetic — never post to the real Confident AI API.
    delete process.env.CONFIDENT_API_KEY;
    _resetWorkerCaseCount();
  });

  afterEach(() => {
    delete process.env[DEEPEVAL_RESULTS_DIR];
    setIsRunningDeepEval(wasRunning);
    if (savedApiKey === undefined) delete process.env.CONFIDENT_API_KEY;
    else process.env.CONFIDENT_API_KEY = savedApiKey;
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("persists and reads back a single-turn case (serializable round-trip)", () => {
    persistCase(llmCase(true));
    const cases = readPersistedCases(dir);
    expect(cases).toHaveLength(1);
    expect(cases[0].conversational).toBe(false);
    expect(cases[0].entry.input).toBe("in");
    expect(cases[0].entry.actualOutput).toBe("out");
    expect(cases[0].metricsData[0].name).toBe("M");
  });

  it("persists a conversational case with turns", () => {
    persistCase({
      testCase: new ConversationalTestCase({
        turns: [new Turn({ role: "user", content: "hi" })],
      }),
      metricsData: [metricData(true)],
      runDuration: 0,
    });
    const cases = readPersistedCases(dir);
    expect(cases[0].conversational).toBe(true);
    expect((cases[0].entry.turns as unknown[]).length).toBe(1);
  });

  it("is a no-op when no results dir is configured", () => {
    delete process.env[DEEPEVAL_RESULTS_DIR];
    persistCase(llmCase(true));
    expect(readPersistedCases(dir)).toHaveLength(0);
  });

  it("merges multiple persisted cases for wrap-up", async () => {
    persistCase(llmCase(true));
    persistCase(llmCase(false));
    expect(readPersistedCases(dir)).toHaveLength(2);
    // No CONFIDENT_API_KEY in tests → posting is a no-op returning nulls.
    const result = await wrapUpTestRun(dir, { printResults: false });
    expect(result).toEqual({ link: null, testRunId: null });
  });

  it("collector persists via assertTest when running under the CLI", async () => {
    setIsRunningDeepEval(true);
    await assertTest(new LLMTestCase({ input: "q", actualOutput: "a" }), [
      new FakeMetric(true),
    ]);
    const cases = readPersistedCases(dir);
    expect(cases).toHaveLength(1);
    expect(cases[0].entry.input).toBe("q");
  });

  it("collector does NOT persist when not running under the CLI", async () => {
    setIsRunningDeepEval(false);
    await assertTest(new LLMTestCase({ input: "q", actualOutput: "a" }), [
      new FakeMetric(true),
    ]);
    expect(readPersistedCases(dir)).toHaveLength(0);
  });
});
