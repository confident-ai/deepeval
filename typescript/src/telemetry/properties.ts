// Port of `deepeval/telemetry/properties.py`.
//
// The wire keys are snake_case -- `eval.test_case_count`, not
// `eval.testCaseCount`. This is the one deliberate exception to the SDK's
// casing: both SDKs write into one PostHog project, and a key differing by
// casing becomes a second, half-populated property. `EventProperties` fields
// stay camelCase, so only `FIELD_TO_PROP` crosses conventions.

import { Entrypoint, Feature } from "@/telemetry/events";

export type PropValue = string | number | boolean | string[];

export enum Prop {
  // telemetry meta
  SCHEMA_VERSION = "telemetry.schema_version",
  SDK_LANGUAGE = "sdk.language",
  SDK_VERSION = "deepeval.version",
  RUNTIME = "runtime.kind",
  // identity
  USER_STATUS = "user.status",
  USER_ID = "user.unique_id",
  LOGGED_IN = "user.logged_in",
  // evaluation
  ENTRYPOINT = "eval.entrypoint",
  RUN_ID = "eval.run_id",
  OUTCOME = "eval.outcome",
  ERROR_TYPE = "eval.error_type",
  TEST_CASE_COUNT = "eval.test_case_count",
  GOLDEN_COUNT = "eval.golden_count",
  TURN_KIND = "eval.turn_kind",
  METRIC_RUNS = "eval.metric_runs",
  METRICS = "eval.metrics",
  METRICS_COUNT = "eval.metrics_count",
  ASYNC_MODE = "eval.async_mode",
  IN_COMPONENT = "eval.in_component",
  FLUSH_REASON = "eval.flush_reason",
  // judge
  PROVIDER = "judge.provider",
  MODEL = "judge.model",
  // tracing
  TRACING_ENABLED = "tracing.enabled",
  TRACED = "tracing.traced",
  TRACE_COUNT = "tracing.trace_count",
  INTEGRATION = "tracing.integration",
  INTEGRATIONS = "tracing.integrations",
  INTEGRATIONS_COUNT = "tracing.integrations_count",
  INTEGRATIONS_PRIMARY = "tracing.integrations_primary",
  // feature adoption
  FEATURE_NAME = "feature.name",
  FEATURE_STATUS = "feature.status",
  LAST_FEATURE = "feature.last",
  // cli and login funnel
  CLI_COMMAND = "cli.command",
  PROMPT_SURFACE = "login.surface",
  LOGIN_OUTCOME = "login.outcome",
  LOGIN_METHOD = "login.method",
  // synthesizer
  SYNTH_METHOD = "synthesizer.method",
  SYNTH_MAX_GENERATIONS = "synthesizer.max_generations",
  SYNTH_NUM_EVOLUTIONS = "synthesizer.num_evolutions",
  SYNTH_EVOLUTIONS = "synthesizer.evolutions",
  // benchmark
  BENCHMARK_NAME = "benchmark.name",
  BENCHMARK_NUM_TASKS = "benchmark.num_tasks",
  // conversation simulator
  NUM_CONVERSATIONS = "simulator.num_conversations",
}

/** Which SDK emitted the event. An absent value means Python. */
export enum Language {
  PYTHON = "python",
  TYPESCRIPT = "typescript",
}

export enum Runtime {
  CI_GITHUB = "ci_github",
  CI_GITLAB = "ci_gitlab",
  CI_OTHER = "ci_other",
  CONTAINER = "container",
  NOTEBOOK = "notebook",
  INTERACTIVE = "interactive",
  SCRIPT = "script",
}

export enum UserStatus {
  NEW = "new",
  OLD = "old",
}

export enum Outcome {
  COMPLETED = "completed",
  ERRORED = "errored",
  INTERRUPTED = "interrupted",
}

/**
 * `MIXED` is unreachable from one `evaluate()` call, which rejects a mixed
 * list, but a test-runner session accumulates across many `toPass` calls.
 */
export enum TurnKind {
  SINGLE_TURN = "single_turn",
  MULTI_TURN = "multi_turn",
  MIXED = "mixed",
}

/**
 * Why a standalone metric batch was sent. A threshold or interval flush is a
 * partial session, so `eval.metric_runs` must be summed, not counted.
 */
export enum FlushReason {
  THRESHOLD = "threshold",
  INTERVAL = "interval",
  PROCESS_EXIT = "process_exit",
  MANUAL = "manual",
}

// Fallback for judge classes the package does not ship. See `judge.ts`.
export const CUSTOM_PROVIDER = "custom";

// Sent instead of a model name outside the known registry, so user-defined
// deployment names cannot leak.
export const UNKNOWN_MODEL = "other";

export const UNKNOWN_CLI_COMMAND = "unknown";

export enum LoginPromptSurface {
  POST_EVAL = "post_eval",
  POST_ARENA = "post_arena",
  CLI_VIEW = "cli_view",
  CLI_VIEW_NO_RUN = "cli_view_no_run",
  INSPECT_TUI = "inspect_tui",
}

export enum LoginOutcome {
  COMPLETED = "completed",
  ABANDONED = "abandoned",
  FAILED = "failed",
}

/** Only the browser flow returns an email, so only it can attach an identity. */
export enum LoginMethod {
  BROWSER = "browser",
  PASTE = "paste",
  API_KEY_FLAG = "api_key_flag",
  UNKNOWN = "unknown",
}

/** Keys of the on-disk identity file. */
export enum TelemetryKey {
  ID = "DEEPEVAL_ID",
  STATUS = "DEEPEVAL_STATUS",
  LAST_FEATURE = "DEEPEVAL_LAST_FEATURE",
  SEEN_FEATURES = "DEEPEVAL_SEEN_FEATURES",
  LOGGED_IN_WITH = "LOGGED_IN_WITH",
}

export interface EventProperties {
  // telemetry meta
  schemaVersion?: number;
  sdkLanguage?: Language;
  sdkVersion?: string;
  runtime?: Runtime;
  // identity
  userStatus?: UserStatus;
  userId?: string;
  loggedIn?: boolean;
  // evaluation
  entrypoint?: Entrypoint;
  runId?: string;
  outcome?: Outcome;
  errorType?: string;
  testCaseCount?: number;
  goldenCount?: number;
  turnKind?: TurnKind;
  metricRuns?: number;
  metrics?: string[];
  metricsCount?: number;
  asyncMode?: boolean;
  inComponent?: boolean;
  flushReason?: FlushReason;
  // judge
  provider?: string;
  model?: string;
  // tracing
  tracingEnabled?: boolean;
  traced?: boolean;
  traceCount?: number;
  integration?: string;
  integrations?: string[];
  integrationsCount?: number;
  integrationsPrimary?: string;
  // feature adoption
  featureName?: Feature;
  featureStatus?: UserStatus;
  lastFeature?: Feature;
  // cli and login funnel
  cliCommand?: string;
  promptSurface?: LoginPromptSurface;
  loginOutcome?: LoginOutcome;
  loginMethod?: LoginMethod;
  // synthesizer
  synthMethod?: string;
  synthMaxGenerations?: number;
  synthNumEvolutions?: number;
  synthEvolutions?: string[];
  // benchmark
  benchmarkName?: string;
  benchmarkNumTasks?: number;
  // conversation simulator
  numConversations?: number;
}

export const FIELD_TO_PROP: Record<keyof EventProperties, Prop> = {
  schemaVersion: Prop.SCHEMA_VERSION,
  sdkLanguage: Prop.SDK_LANGUAGE,
  sdkVersion: Prop.SDK_VERSION,
  runtime: Prop.RUNTIME,
  userStatus: Prop.USER_STATUS,
  userId: Prop.USER_ID,
  loggedIn: Prop.LOGGED_IN,
  entrypoint: Prop.ENTRYPOINT,
  runId: Prop.RUN_ID,
  outcome: Prop.OUTCOME,
  errorType: Prop.ERROR_TYPE,
  testCaseCount: Prop.TEST_CASE_COUNT,
  goldenCount: Prop.GOLDEN_COUNT,
  turnKind: Prop.TURN_KIND,
  metricRuns: Prop.METRIC_RUNS,
  metrics: Prop.METRICS,
  metricsCount: Prop.METRICS_COUNT,
  asyncMode: Prop.ASYNC_MODE,
  inComponent: Prop.IN_COMPONENT,
  flushReason: Prop.FLUSH_REASON,
  provider: Prop.PROVIDER,
  model: Prop.MODEL,
  tracingEnabled: Prop.TRACING_ENABLED,
  traced: Prop.TRACED,
  traceCount: Prop.TRACE_COUNT,
  integration: Prop.INTEGRATION,
  integrations: Prop.INTEGRATIONS,
  integrationsCount: Prop.INTEGRATIONS_COUNT,
  integrationsPrimary: Prop.INTEGRATIONS_PRIMARY,
  featureName: Prop.FEATURE_NAME,
  featureStatus: Prop.FEATURE_STATUS,
  lastFeature: Prop.LAST_FEATURE,
  cliCommand: Prop.CLI_COMMAND,
  promptSurface: Prop.PROMPT_SURFACE,
  loginOutcome: Prop.LOGIN_OUTCOME,
  loginMethod: Prop.LOGIN_METHOD,
  synthMethod: Prop.SYNTH_METHOD,
  synthMaxGenerations: Prop.SYNTH_MAX_GENERATIONS,
  synthNumEvolutions: Prop.SYNTH_NUM_EVOLUTIONS,
  synthEvolutions: Prop.SYNTH_EVOLUTIONS,
  benchmarkName: Prop.BENCHMARK_NAME,
  benchmarkNumTasks: Prop.BENCHMARK_NUM_TASKS,
  numConversations: Prop.NUM_CONVERSATIONS,
};

/** Drop absent fields and rewrite the rest onto their wire keys. */
export function toRecord(
  properties: EventProperties,
): Record<string, PropValue> {
  const payload: Record<string, PropValue> = {};
  for (const [field, prop] of Object.entries(FIELD_TO_PROP)) {
    const value = properties[field as keyof EventProperties];
    if (value === undefined || value === null) continue;
    payload[prop] = value as PropValue;
  }
  return payload;
}

export function mergedWith(
  base: EventProperties,
  other: EventProperties,
): Record<string, PropValue> {
  return { ...toRecord(base), ...toRecord(other) };
}
