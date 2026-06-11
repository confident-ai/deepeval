import { MultiBar, type SingleBar, Presets } from "cli-progress";
import {
  BaseMetric,
  BaseMetricCore,
  BaseConversationalMetric,
} from "../metrics";
import {
  LLMTestCase,
  ConversationalTestCase,
  resolveRetrievalContext,
} from "../test-case";
import { MissingTestCaseParamsError } from "../errors";
import { TestResult, MetricData, EvaluationResult } from "./types";
import {
  AsyncConfig,
  DisplayConfig,
  ErrorConfig,
  CacheConfig,
  DEFAULT_ASYNC_CONFIG,
  DEFAULT_DISPLAY_CONFIG,
  DEFAULT_ERROR_CONFIG,
  DEFAULT_CACHE_CONFIG,
} from "./configs";
import {
  printResultsTable,
  printHyperparametersWarning,
  printCompletionSummary,
  exportToMarkdown,
} from "./console-report";
import { postTestRun } from "./confident";
import { type EvaluatedCase } from "./types";

type AnyTestCase = LLMTestCase | ConversationalTestCase;
type AnyMetric = BaseMetric | BaseConversationalMetric;

export interface EvaluateOptions {
  asyncConfig?: AsyncConfig;
  displayConfig?: DisplayConfig;
  errorConfig?: ErrorConfig;
  cacheConfig?: CacheConfig;
  /** Mark this as the official test run for the dataset on Confident AI. */
  official?: boolean;
}

/** A conversational metric runs on a `ConversationalTestCase`; otherwise single-turn. */
function metricMatchesCase(metric: AnyMetric, testCase: AnyTestCase): boolean {
  const caseIsConversational = testCase instanceof ConversationalTestCase;
  const metricIsConversational = metric instanceof BaseConversationalMetric;
  return caseIsConversational === metricIsConversational;
}

/**
 * Run `metrics` over `testCases` and collect the results. Accepts both
 * single-turn (`LLMTestCase` + `BaseMetric`) and conversational
 * (`ConversationalTestCase` + `BaseConversationalMetric`) inputs — each metric
 * only runs on the test-case type it matches.
 *
 * TS port of Python's `evaluate()`. Posts results to Confident AI as a TestRun
 * when logged in. Caching + cross-test-case concurrency are still placeholders.
 */
export async function evaluate(
  testCases: AnyTestCase[],
  metrics: AnyMetric[],
  options: EvaluateOptions = {},
): Promise<EvaluationResult> {
  const display: Required<DisplayConfig> = {
    ...DEFAULT_DISPLAY_CONFIG,
    ...options.displayConfig,
  };
  const errorCfg: Required<ErrorConfig> = {
    ...DEFAULT_ERROR_CONFIG,
    ...options.errorConfig,
  };
  // Reserved for when concurrency/caching are wired up.
  void ({ ...DEFAULT_ASYNC_CONFIG, ...options.asyncConfig } as AsyncConfig);
  void ({ ...DEFAULT_CACHE_CONFIG, ...options.cacheConfig } as CacheConfig);

  // Per-case work list: only the metrics that match each case's type.
  const work = testCases.map((testCase, index) => ({
    index,
    testCase,
    metrics: metrics.filter((m) => metricMatchesCase(m, testCase)),
  }));
  const total = work.reduce((sum, w) => sum + w.metrics.length, 0);

  // Print each metric's description line once (mirrors Python's evaluate()).
  if (display.showIndicator) {
    for (const metric of metrics) {
      process.stderr.write(metric.describe() + "\n");
    }
  }

  // A nested progress: a top "Evaluating N test case(s)" bar + one
  // "🎯 Evaluating test case #i" bar per case (mirrors Python's rich progress).
  let multibar: MultiBar | null = null;
  let mainBar: SingleBar | null = null;
  let caseBars: SingleBar[] = [];
  if (display.showIndicator && total > 0) {
    const PURPLE = "\x1b[38;2;106;0;255m"; // filled
    const TRACK = "\x1b[38;2;70;70;82m"; // dim grey (empty track)
    const RESET = "\x1b[0m";
    const BARSIZE = 40;
    multibar = new MultiBar(
      {
        format: "{label} {bar} {percentage}% {duration_formatted}",
        // formatBar's return is used verbatim for {bar} (no string-slicing), so
        // per-char ANSI is safe here. Filled = bright purple, track = dim purple.
        formatBar: (progress, options) => {
          const size = options.barsize ?? BARSIZE;
          const filled = Math.round(progress * size);
          return (
            PURPLE +
            "━".repeat(filled) +
            TRACK +
            "━".repeat(size - filled) +
            RESET
          );
        },
        hideCursor: true,
        clearOnComplete: true,
        stream: process.stderr,
        barsize: BARSIZE,
      },
      Presets.shades_classic,
    );
    mainBar = multibar.create(work.length, 0, {
      label: `Evaluating ${work.length} test case(s)`,
    });
    caseBars = work.map((w, i) =>
      multibar!.create(Math.max(w.metrics.length, 1), 0, {
        label: `    🎯 Evaluating test case #${i}`,
      }),
    );
  }

  // During batch runs we show the batch bars, not each metric's own spinner.
  const originalShowIndicator = metrics.map((m) => m.showIndicator);
  metrics.forEach((m) => {
    m.showIndicator = false;
    if (options.displayConfig?.verboseMode !== undefined) {
      m.verboseMode = options.displayConfig.verboseMode;
    }
  });

  const testResults: TestResult[] = [];
  const evaluatedCases: EvaluatedCase[] = [];
  const startTime = Date.now();
  try {
    // Test cases run sequentially (metric instances are stateful and reused);
    // matching metrics within a test case run concurrently (distinct instances).
    // TODO: parallelize across test cases by cloning metrics per case, honoring maxConcurrent.
    for (const { index, testCase, metrics: applicable } of work) {
      const caseBar = caseBars[index];
      const caseStart = Date.now();
      const metricsData = await Promise.all(
        applicable.map((m) =>
          runMetric(m, testCase, errorCfg, () => caseBar?.increment()),
        ),
      );
      caseBar?.update(Math.max(applicable.length, 1));
      mainBar?.increment();
      testResults.push(buildTestResult(index, testCase, metricsData));
      evaluatedCases.push({
        testCase,
        metricsData,
        runDuration: (Date.now() - caseStart) / 1000,
      });
    }
  } finally {
    multibar?.stop();
    metrics.forEach((m, i) => {
      m.showIndicator = originalShowIndicator[i];
    });
  }
  const runDuration = (Date.now() - startTime) / 1000;

  if (display.printResults) {
    printResultsTable(testResults, {
      truncatePassing: display.truncatePassingCases,
    });
    printHyperparametersWarning();
  }

  // Optionally write the report to a Markdown/MDX file.
  if (display.fileOutputDir) {
    exportToMarkdown(testResults, display.fileOutputDir, display.fileType);
  }

  // Post results to Confident AI (no-op + returns nulls unless logged in).
  const { link, testRunId } = await postTestRun(
    evaluatedCases,
    runDuration,
    options.official ?? false,
  );

  if (display.printResults && !link) {
    const tokenCost = testResults
      .flatMap((t) => t.metricsData ?? [])
      .reduce((sum, m) => sum + (m.evaluationCost ?? 0), 0);
    const passed = testResults.filter((t) => t.success).length;
    printCompletionSummary({
      runDuration,
      tokenCost,
      passed,
      failed: testResults.length - passed,
    });
  }

  return { testResults, confidentLink: link, testRunId };
}

export async function runMetric(
  metric: AnyMetric,
  testCase: AnyTestCase,
  errorCfg: Required<ErrorConfig>,
  onDone: () => void,
): Promise<MetricData> {
  // fresh state per (metric, test case)
  metric.score = undefined;
  metric.success = undefined;
  metric.reason = undefined;
  metric.error = undefined;
  metric.skipped = false;

  try {
    // Dispatched in `evaluate`, so the metric matches the test case type.
    await (metric.measure as (tc: AnyTestCase) => Promise<number>)(testCase);
  } catch (e) {
    if (e instanceof MissingTestCaseParamsError && errorCfg.skipOnMissingParams) {
      metric.skipped = true;
    } else if (errorCfg.ignoreErrors) {
      metric.error = (e as Error).message;
    } else {
      onDone();
      throw e;
    }
  }
  onDone();
  return buildMetricData(metric);
}

function buildMetricData(metric: BaseMetricCore): MetricData {
  return {
    name: metric.name,
    threshold: metric.threshold,
    success: metric.skipped ? true : (metric.success ?? false),
    score: metric.score,
    reason: metric.reason,
    strictMode: metric.strictMode,
    evaluationModel: metric.evaluationModel,
    evaluationCost: metric.evaluationCost,
    verboseLogs: metric.verboseLogs,
    error: metric.error,
    skipped: metric.skipped,
  };
}

export function buildTestResult(
  index: number,
  testCase: AnyTestCase,
  metricsData: MetricData[],
): TestResult {
  const success = metricsData.every((m) => m.skipped || m.success);

  if (testCase instanceof ConversationalTestCase) {
    return {
      name: testCase.name ?? `test_case_${index}`,
      success,
      metricsData,
      conversational: true,
      index,
      turns: testCase.turns,
    };
  }

  return {
    name: testCase.name ?? `test_case_${index}`,
    success,
    metricsData,
    conversational: false,
    index,
    input: testCase.input,
    actualOutput: testCase.actualOutput,
    expectedOutput: testCase.expectedOutput,
    context: testCase.context,
    retrievalContext: resolveRetrievalContext(testCase.retrievalContext),
  };
}
