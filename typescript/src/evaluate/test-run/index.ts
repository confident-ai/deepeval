export {
  runMetrics,
  runTestCaseMetrics,
  runTraceMetrics,
  evaluateCase,
  type MetricsOutcome,
  type ToPassTarget,
} from "./run-metrics";
export { buildFailureMessage } from "./errors";
export { globalResultCollector } from "./collector";
export {
  persistCase,
  readPersistedCases,
  wrapUpTestRun,
  getResultsDir,
  type WrapUpOptions,
} from "./store";
export {
  beginTraceCapture,
  endTraceCapture,
  isCapturingTraces,
  getCapturedTraces,
  getLatestCapturedTrace,
} from "./trace-scope";
