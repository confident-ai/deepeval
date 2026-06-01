import { Environment } from "../tracing/utils";

export interface Settings {
  CONFIDENT_TRACE_ENVIRONMENT?: Environment;
  CONFIDENT_TRACE_VERBOSE?: boolean;
  CONFIDENT_TRACE_SAMPLE_RATE?: number;
  CONFIDENT_OTEL_URL?: string;
}

let _settings_singleton: Settings | null = null;

export function getSettings(): Settings {
  if (_settings_singleton === null) {
    _settings_singleton = {
      CONFIDENT_TRACE_ENVIRONMENT:
        (process.env.CONFIDENT_TRACE_ENVIRONMENT as Environment) ||
        Environment.DEVELOPMENT,

      CONFIDENT_TRACE_VERBOSE:
        process.env.CONFIDENT_TRACE_VERBOSE !== undefined
          ? ["yes", "true", "1"].includes(
              process.env.CONFIDENT_TRACE_VERBOSE.toLowerCase(),
            )
          : true,

      CONFIDENT_TRACE_SAMPLE_RATE:
        process.env.CONFIDENT_TRACE_SAMPLE_RATE !== undefined
          ? parseFloat(process.env.CONFIDENT_TRACE_SAMPLE_RATE)
          : 1.0,

      CONFIDENT_OTEL_URL:
        (process.env.CONFIDENT_OTEL_URL as string) ||
        "https://otel.confident-ai.com",
    };
  }

  return _settings_singleton;
}

export function resetSettings({
  reloadDotenv = false,
}: { reloadDotenv?: boolean } = {}): Settings {
  if (reloadDotenv) {
    // TODO
  }
  _settings_singleton = null;
  return getSettings();
}
