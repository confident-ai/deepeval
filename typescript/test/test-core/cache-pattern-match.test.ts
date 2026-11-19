import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { CACHE_FILE, HIDDEN_DIR } from "@/constants";
import { runMetric } from "@/evaluate/evaluate";
import {
  clearCache,
  flushCache,
  testCaseCacheKey,
} from "@/evaluate/test-run/cache";
import { BaseMetric } from "@/metrics/base-metrics";
import { PatternMatchMetric } from "@/metrics/pattern-match/pattern-match";
import { LLMTestCase } from "@/test-case";

// Always keep cache files beneath the temporary working directory, even when
// the developer has configured a different DEEPEVAL_CACHE_FOLDER.
jest.mock("@/constants", () => ({
  ...jest.requireActual("@/constants"),
  HIDDEN_DIR: ".deepeval",
}));

// Metric execution and cache I/O are real; anonymous telemetry is unrelated to
// this regression and must not send events from the unit tests.
jest.mock("@/telemetry", () => ({
  ...jest.requireActual("@/telemetry"),
  recordMetric: jest.fn(),
}));

const run = (testCase: LLMTestCase, metric: BaseMetric) =>
  runMetric(
    metric,
    testCase,
    { ignoreErrors: false, skipOnMissingParams: false },
    () => {},
    { useCache: true, writeCache: true },
  );

const patternMetric = (pattern: string, ignoreCase?: boolean) =>
  new PatternMatchMetric({ pattern, ignoreCase, showIndicator: false });

class FixedMetric extends BaseMetric {
  constructor() {
    super(1);
  }

  get name(): string {
    return "Fixed";
  }

  measure(): number {
    this.score = 1;
    this.success = true;
    return this.score;
  }
}

describe("PatternMatchMetric result caching", () => {
  let directory: string;
  let cacheFile: string;

  beforeEach(() => {
    directory = fs.mkdtempSync(
      path.join(os.tmpdir(), "deepeval-pattern-cache-"),
    );
    jest.spyOn(process, "cwd").mockReturnValue(directory);
    jest.replaceProperty(process, "env", {
      ...process.env,
      DEEPEVAL_FILE_SYSTEM: "",
      CONFIDENT_TRACE_INTERNAL: "false",
    });
    cacheFile = path.join(directory, HIDDEN_DIR, CACHE_FILE);
    clearCache();
  });

  afterEach(() => {
    clearCache();
    jest.restoreAllMocks();
    fs.rmSync(directory, { recursive: true, force: true });
  });

  // clearCache resets the in-memory read cache as well as deleting the file.
  // Restore its serialized contents so the next run must read them from disk.
  function reloadFromDisk(contents = fs.readFileSync(cacheFile, "utf-8")) {
    clearCache();
    fs.writeFileSync(cacheFile, contents, "utf-8");
  }

  async function seed(testCase: LLMTestCase, metric: BaseMetric) {
    const result = await run(testCase, metric);
    flushCache();
    reloadFromDisk();
    return result;
  }

  it("rechecks a passing output when the pattern is tightened", async () => {
    const testCase = new LLMTestCase({
      input: "Order id?",
      actualOutput: "I don't know",
    });
    expect(await seed(testCase, patternMetric(".*"))).toMatchObject({
      score: 1,
      success: true,
    });

    const strict = patternMetric("\\d+");
    const measure = jest.spyOn(strict, "measure");
    expect(await run(testCase, strict)).toMatchObject({
      score: 0,
      success: false,
    });
    expect(measure).toHaveBeenCalledTimes(1);
  });

  it("rechecks a passing output when case-insensitive matching is disabled", async () => {
    const testCase = new LLMTestCase({
      input: "Status?",
      actualOutput: "APPROVED",
    });
    expect(await seed(testCase, patternMetric("approved", true))).toMatchObject(
      {
        score: 1,
        success: true,
      },
    );

    const strict = patternMetric("approved", false);
    const measure = jest.spyOn(strict, "measure");
    expect(await run(testCase, strict)).toMatchObject({
      score: 0,
      success: false,
    });
    expect(measure).toHaveBeenCalledTimes(1);
  });

  it.each([false, true])(
    "reuses the same pattern and ignoreCase=%s after a disk round trip",
    async (ignoreCase) => {
      const testCase = new LLMTestCase({
        input: "Status?",
        actualOutput: "approved",
      });
      const cached = await seed(
        testCase,
        patternMetric("approved", ignoreCase),
      );
      const same = patternMetric("approved", ignoreCase);
      const measure = jest.spyOn(same, "measure");

      expect(await run(testCase, same)).toEqual(cached);
      expect(measure).not.toHaveBeenCalled();
    },
  );

  it.each([
    [undefined, false],
    [false, undefined],
  ])(
    "treats ignoreCase=%s and ignoreCase=%s as equivalent",
    async (first, next) => {
      const testCase = new LLMTestCase({
        input: "Status?",
        actualOutput: "APPROVED",
      });
      const cached = await seed(testCase, patternMetric("approved", first));
      expect(cached).toMatchObject({ score: 0, success: false });
      const same = patternMetric("approved", next);
      const measure = jest.spyOn(same, "measure");

      expect(await run(testCase, same)).toEqual(cached);
      expect(measure).not.toHaveBeenCalled();
    },
  );

  it("keeps distinct pattern configurations available after flushing", async () => {
    const testCase = new LLMTestCase({
      input: "Order id?",
      actualOutput: "unknown",
    });
    await seed(testCase, patternMetric(".*"));
    await run(testCase, patternMetric("\\d+"));
    flushCache();
    reloadFromDisk();

    for (const [pattern, score] of [
      [".*", 1],
      ["\\d+", 0],
    ] as const) {
      const metric = patternMetric(pattern);
      const measure = jest.spyOn(metric, "measure");
      expect(await run(testCase, metric)).toMatchObject({ score });
      expect(measure).not.toHaveBeenCalled();
    }
  });

  it("recomputes legacy PatternMatch entries without pattern settings", async () => {
    const testCase = new LLMTestCase({
      input: "Order id?",
      actualOutput: "unknown",
    });
    await seed(testCase, patternMetric(".*"));
    const legacy = JSON.parse(fs.readFileSync(cacheFile, "utf-8"));
    const configuration =
      legacy.testCases[testCaseCacheKey(testCase)][0].configuration;
    delete configuration.pattern;
    delete configuration.ignoreCase;
    reloadFromDisk(JSON.stringify(legacy));

    const strict = patternMetric("\\d+");
    const measure = jest.spyOn(strict, "measure");
    expect(await run(testCase, strict)).toMatchObject({
      score: 0,
      success: false,
    });
    expect(measure).toHaveBeenCalledTimes(1);
  });

  it("keeps legacy entries reusable for metrics without pattern settings", async () => {
    const testCase = new LLMTestCase({
      input: "Question",
      actualOutput: "Answer",
    });
    const cached = await seed(testCase, new FixedMetric());
    const legacy = JSON.parse(fs.readFileSync(cacheFile, "utf-8"));
    const configuration =
      legacy.testCases[testCaseCacheKey(testCase)][0].configuration;
    delete configuration.pattern;
    delete configuration.ignoreCase;
    reloadFromDisk(JSON.stringify(legacy));

    const same = new FixedMetric();
    const measure = jest.spyOn(same, "measure");
    expect(await run(testCase, same)).toEqual(cached);
    expect(measure).not.toHaveBeenCalled();
  });

  it("distinguishes an empty pattern from one that accepts nonempty output", async () => {
    const testCase = new LLMTestCase({
      input: "Question",
      actualOutput: "Answer",
    });
    await seed(testCase, patternMetric(".*"));
    const empty = patternMetric("");
    const measure = jest.spyOn(empty, "measure");

    expect(await run(testCase, empty)).toMatchObject({
      score: 0,
      success: false,
    });
    expect(measure).toHaveBeenCalledTimes(1);
  });
});
