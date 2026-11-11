// Regression tests for the metric-configuration whitelist in
// `evaluate/test-run/cache.ts`, mirroring the shape of #3579's Python tests
// (tests/test_core/test_run/test_cache_metric_configuration.py): fields that
// change the score must split cache entries; irrelevant fields must not.
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { GEval } from "@/metrics/g-eval/g-eval";
import { LLMTestCase } from "@/test-case";
import { DeepEvalBaseLLM } from "@/models";
import {
  cacheMetricData,
  getCachedMetricData,
  flushCache,
  clearCache,
} from "@/evaluate/test-run/cache";
import type { MetricData } from "@/evaluate/types";

// Mirrors #3579's static judge: the cache comparison must not call the judge.
class StaticJudge extends DeepEvalBaseLLM {
  getModelName() {
    return "static-judge";
  }
  async generate(): Promise<never> {
    throw new Error("the cache comparison must not call the judge");
  }
  async a_generate(): Promise<never> {
    throw new Error("the cache comparison must not call the judge");
  }
}

const judge = new StaticJudge();
const testCase = new LLMTestCase({ input: "Hi", actualOutput: "Hello" });

const metricData = (metric: GEval): MetricData => ({
  name: metric.name,
  threshold: null,
  success: true,
  score: 0.7,
  reason: "cached score",
  strictMode: false,
  flaky: false,
  evaluationModel: metric.evaluationModel,
  skipped: false,
});

const geval = (overrides: Partial<ConstructorParameters<typeof GEval>[0]>) =>
  new GEval({
    name: "Refund handling",
    criteria: "Does the reply handle the refund request well?",
    model: judge,
    ...overrides,
  } as ConstructorParameters<typeof GEval>[0]);

const cacheFor = (metric: GEval) => {
  cacheMetricData(testCase, metric, metricData(metric));
  flushCache();
};

const scratch = path.join(os.tmpdir(), "cache-metric-configuration-cwd");

beforeAll(() => {
  fs.mkdirSync(scratch, { recursive: true });
  process.chdir(scratch);
});
afterAll(() => {
  process.chdir(__dirname);
  fs.rmSync(scratch, { recursive: true, force: true });
});
beforeEach(() => clearCache());
afterEach(() => clearCache());

test("changed topLogprobs does not reuse cached score", () => {
  cacheFor(geval({ topLogprobs: 20 }));

  expect(getCachedMetricData(testCase, geval({ topLogprobs: 1 }))).toBeNull();
});

test("changed evaluationTemplate does not reuse cached score", () => {
  cacheFor(geval({}));

  const overridden = geval({
    evaluationTemplate: {
      generateEvaluationResults: (vars, renderDefault) =>
        renderDefault(vars) + "\n\nExtra rule that changes the judge prompt.",
    },
  });
  expect(getCachedMetricData(testCase, overridden)).toBeNull();
});

test("same topLogprobs and evaluationTemplate reuse cached score", () => {
  cacheFor(geval({ topLogprobs: 3 }));

  const hit = getCachedMetricData(testCase, geval({ topLogprobs: 3 }));
  expect(hit).not.toBeNull();
  expect(hit?.score).toBe(0.7);
});

test("irrelevant fields (showIndicator, verboseMode) do not split the cache", () => {
  cacheFor(geval({}));

  expect(
    getCachedMetricData(testCase, geval({ showIndicator: true })),
  ).not.toBeNull();
  expect(
    getCachedMetricData(testCase, geval({ verboseMode: true })),
  ).not.toBeNull();
});

test("cache files written before these fields were recorded do not reuse", () => {
  // Older entries have no topLogprobs/evaluationTemplate in the stored
  // configuration. Reusing them would risk a stale score, so they miss.
  cacheFor(geval({}));

  const file = path.join(process.cwd(), ".deepeval", ".deepeval-cache.json");
  const onDisk = JSON.parse(fs.readFileSync(file, "utf-8"));
  for (const entries of Object.values<any[]>(onDisk.testCases)) {
    for (const entry of entries) {
      delete entry.configuration.topLogprobs;
      delete entry.configuration.evaluationTemplate;
    }
  }
  fs.writeFileSync(file, JSON.stringify(onDisk));

  jest.resetModules();
  // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
  const freshCache = require("@/evaluate/test-run/cache");
  expect(freshCache.getCachedMetricData(testCase, geval({}))).toBeNull();
});
