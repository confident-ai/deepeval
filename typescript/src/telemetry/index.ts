// Anonymous telemetry, in parity with `deepeval/telemetry/` in the Python SDK:
// the same event names, property keys, and identity file. Both SDKs report into
// one PostHog project, told apart only by `sdk.language`.
//
// Opt out with `DEEPEVAL_TELEMETRY_OPT_OUT=1`.

export {
  LoginSpan,
  beginEvaluationRun,
  captureBenchmarkRun,
  captureCliCommand,
  captureConversationSimulatorRun,
  captureEvaluationRun,
  captureLoginEvent,
  captureLoginPromptShown,
  captureLoginPromptShownOnce,
  captureSynthesizerRun,
  recordLoginCompleted,
  recordTracingIntegration,
  type EvaluationRunOptions,
  type SynthesizerRunOptions,
} from "@/telemetry/api";

export {
  NoopBackend,
  PostHogBackend,
  baseProperties,
  capture,
  flush,
  getBackend,
  installedIntegrations,
  registerIntegration,
  setBackend,
  telemetryOptOut,
  type TelemetryBackend,
} from "@/telemetry/client";

export {
  RunAccumulator,
  STANDALONE_FLUSH_INTERVAL_MS,
  STANDALONE_FLUSH_THRESHOLD,
  StandaloneAccumulator,
  currentRun,
  flushStandaloneMetrics,
  inComponentScope,
  pushAmbientRun,
  recordGolden,
  recordMetric,
  recordTestCase,
  resetForTesting,
  turnKindOf,
  withComponentScope,
  withRun,
  type RecordMetricOptions,
} from "@/telemetry/context";

export {
  TELEMETRY_RUN_ID_ENV_VAR,
  TELEMETRY_SCHEMA_VERSION,
  Entrypoint,
  Event,
  Feature,
} from "@/telemetry/events";

export {
  TELEMETRY_DATA_FILE,
  getFeatureStatus,
  getIdentity,
  getLastFeature,
  getLoggedInWith,
  getStatus,
  getUniqueId,
  isLoggedIn,
  readTelemetryFile,
  setLastFeature,
  setLoggedInWith,
  telemetryPath,
  writeTelemetryFile,
  type Identity,
} from "@/telemetry/identity";

export { describeJudge, type JudgeDescription } from "@/telemetry/judge";

export {
  CUSTOM_PROVIDER,
  FIELD_TO_PROP,
  FlushReason,
  Language,
  LoginMethod,
  LoginOutcome,
  LoginPromptSurface,
  Outcome,
  Prop,
  Runtime,
  TelemetryKey,
  TurnKind,
  UNKNOWN_CLI_COMMAND,
  UNKNOWN_MODEL,
  UserStatus,
  mergedWith,
  toRecord,
  type EventProperties,
  type PropValue,
} from "@/telemetry/properties";

export { detectRuntime } from "@/telemetry/runtime";
