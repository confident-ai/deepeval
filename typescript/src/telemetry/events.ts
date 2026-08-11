// Port of `deepeval/telemetry/events.py`. Every value here must match its
// Python counterpart exactly: both SDKs report into one PostHog project, so a
// rename on one side forks a series rather than failing.

export const TELEMETRY_SCHEMA_VERSION = 2;

// Carries one run's id into Vitest workers, which are fresh processes.
export const TELEMETRY_RUN_ID_ENV_VAR = "DEEPEVAL_TELEMETRY_RUN_ID";

export enum Event {
  EVALUATION = "Evaluation",
  SYNTHESIZER = "Synthesizer",
  CONVERSATION_SIMULATOR = "Conversation Simulator",
  BENCHMARK = "Benchmark",
  INTEGRATION_INSTALLED = "Integration Installed",
  CLI_COMMAND = "CLI Command",
  LOGIN_PROMPT_SHOWN = "Login Prompt Shown",
  LOGIN = "Login",
}

/** A dimension on `Event.EVALUATION` rather than one event name each. */
export enum Entrypoint {
  EVALUATE = "evaluate",
  EVALS_ITERATOR = "evals_iterator",
  /** Python's test runner. Never emitted from here. */
  PYTEST = "pytest",
  VITEST = "vitest",
  COMPARE = "compare",
  STANDALONE = "standalone",
}

export enum Feature {
  REDTEAMING = "redteaming",
  SYNTHESIZER = "synthesizer",
  EVALUATION = "evaluation",
  COMPONENT_EVALUATION = "component_evaluation",
  BENCHMARK = "benchmark",
  CONVERSATION_SIMULATOR = "conversation_simulator",
  TRACING_INTEGRATION = "tracing_integration",
  UNKNOWN = "unknown",
}
