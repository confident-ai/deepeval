export {
  runMetrics,
  runTestCaseMetrics,
  runCallbackMetrics,
  evaluateCase,
  type MetricsOutcome,
  type ToPassTarget,
  type ToPassCallback,
  type ToPassTask,
  type ToPassOptions,
} from "@/evaluate/test-run/run-metrics";
export { buildFailureMessage } from "@/evaluate/test-run/errors";
export { globalResultCollector } from "@/evaluate/test-run/collector";
export {
  persistCase,
  readPersistedCases,
  wrapUpTestRun,
  getResultsDir,
  type WrapUpOptions,
} from "@/evaluate/test-run/store";
export {
  saveLatestTestRun,
  readLatestTestRun,
  deleteLatestTestRun,
  exportTestRunJson,
  type LocalTestRun,
} from "@/evaluate/test-run/local";
export {
  beginTraceCapture,
  endTraceCapture,
  isCapturingTraces,
  getCapturedTraces,
  getLatestCapturedTrace,
  collectTracesFrom,
} from "@/evaluate/test-run/trace-scope";
