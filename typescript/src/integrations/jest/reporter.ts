/**
 * DeepEval Jest Reporter
 *
 * A custom Jest reporter that collects test results and flushes them
 * to the DeepEval backend (Confident AI) after the full test suite runs.
 *
 * Usage — jest.config.js:
 *   module.exports = {
 *     reporters: [
 *       'default',
 *       ['deepeval/integrations/jest/reporter', { apiKey: process.env.DEEPEVAL_API_KEY }],
 *     ],
 *   };
 *
 * Compatible with: Jest 27+, ts-jest, @jest/reporters
 * Depends on: evaluate.ts EvaluationResult shape (no assertTest/TestRunManager required)
 */

import type {
  AggregatedResult,
  TestResult as JestTestResult,
  AssertionResult,
  Reporter,
  ReporterOnStartOptions,
  Test,
} from '@jest/reporters';
import type { Config } from '@jest/types';
import { evaluate } from '../evaluate';
import type { EvaluationResult, TestResult } from '../evaluate/types';

export interface DeepEvalReporterOptions {
  /** Confident AI API key. Falls back to DEEPEVAL_API_KEY env var. */
  apiKey?: string;
  /** If true, failures in the DeepEval flush do NOT fail the Jest run. Default: false */
  silent?: boolean;
  /** Optional run name shown in the Confident AI dashboard */
  runName?: string;
}

/**
 * Extracts DeepEval metric results attached by assertTest() calls.
 * assertTest() annotates the Jest test result via jest.expect.extend or
 * stores results on `global.__deepeval_results__` keyed by test full name.
 */
function extractDeepEvalResult(
  result: AssertionResult
): TestResult | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const store = (global as any).__deepeval_results__ as
    | Map<string, TestResult>
    | undefined;
  if (!store) return null;
  return store.get(result.fullName) ?? null;
}

export class DeepEvalJestReporter implements Reporter {
  private readonly options: Required<DeepEvalReporterOptions>;
  private collectedResults: TestResult[] = [];

  constructor(
    _globalConfig: Config.GlobalConfig,
    options: DeepEvalReporterOptions = {}
  ) {
    this.options = {
      apiKey: options.apiKey ?? process.env.DEEPEVAL_API_KEY ?? '',
      silent: options.silent ?? false,
      runName: options.runName ?? `jest-run-${new Date().toISOString()}`,
    };

    if (!this.options.apiKey && !this.options.silent) {
      console.warn(
        '[DeepEval] No API key found. Set DEEPEVAL_API_KEY or pass apiKey in reporter options.'
      );
    }
  }

  // Called by Jest at the start of each suite — no-op, required by Reporter interface
  onRunStart(
    _results: AggregatedResult,
    _options: ReporterOnStartOptions
  ): void {
    this.collectedResults = [];
  }

  // Called after each test file completes
  onTestResult(
    _test: Test,
    testResult: JestTestResult
  ): void {
    for (const assertion of testResult.testResults) {
      const deepEvalResult = extractDeepEvalResult(assertion);
      if (deepEvalResult) {
        this.collectedResults.push(deepEvalResult);
      }
    }
  }

  // Called once after the entire suite — flush results to Confident AI
  async onRunComplete(
    _contexts: Set<unknown>,
    _results: AggregatedResult
  ): Promise<void> {
    if (this.collectedResults.length === 0) {
      if (!this.options.silent) {
        console.log(
          '[DeepEval] No DeepEval-instrumented tests found. Skipping upload.'
        );
      }
      return;
    }

    if (!this.options.apiKey) {
      if (!this.options.silent) {
        console.warn(
          `[DeepEval] Skipping upload — no API key. ${this.collectedResults.length} result(s) collected but not sent.`
        );
      }
      return;
    }

    try {
      const evaluationResult: EvaluationResult = await evaluate(
        this.collectedResults,
        [],
        {
          apiKey: this.options.apiKey,
          runName: this.options.runName,
        }
      );

      console.log(
        `[DeepEval] ${this.collectedResults.length} test result(s) uploaded.`
      );

      if (evaluationResult.confidentLink) {
        console.log(`[DeepEval] View results: ${evaluationResult.confidentLink}`);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (this.options.silent) {
        // swallow — don't block CI
        console.warn(`[DeepEval] Upload failed (silent mode): ${msg}`);
      } else {
        throw new Error(`[DeepEval] Failed to upload results: ${msg}`);
      }
    }
  }

  // Required stub — Jest calls this to check if the reporter needs forced exit
  getLastError(): Error | void {
    return undefined;
  }
}

export default DeepEvalJestReporter;
