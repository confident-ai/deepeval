const SENSITIVE_KEY =
  /api.?key|authorization|token|secret|password|email|path|prompt|command|file/i;
const SECRET_VALUE =
  /(Bearer\s+\S+|sk-[A-Za-z0-9_-]+|[A-Za-z]+_[A-Za-z0-9_-]{24,})/i;

export interface TelemetryEvent {
  event:
    | "wizard_started"
    | "authentication_started"
    | "authentication_completed"
    | "setup_started"
    | "setup_completed"
    | "setup_failed";
  step?:
    | "bootstrap"
    | "authentication"
    | "environment_detection"
    | "dependency_installation"
    | "configuration"
    | "evaluation";
  result?: "started" | "succeeded" | "failed" | "cancelled";
  errorCode?:
    | "unknown"
    | "network"
    | "permission_denied"
    | "invalid_configuration"
    | "command_failed"
    | "timeout";
  elapsedMs?: number;
}

type TelemetryErrorCode = NonNullable<TelemetryEvent["errorCode"]>;

/** Map a thrown error onto a coarse code, never onto its message. */
export const classifyErrorCode = (error: unknown): TelemetryErrorCode => {
  const message = (
    error instanceof Error ? error.message : String(error)
  ).toLowerCase();
  if (/timed out|timeout|etimedout/.test(message)) return "timeout";
  if (/permission|forbidden|denied|eacces|eperm/.test(message)) {
    return "permission_denied";
  }
  if (/fetch failed|network|socket|enotfound|econnrefused|dns/.test(message)) {
    return "network";
  }
  if (/exited with|spawn|enoent/.test(message)) return "command_failed";
  if (/invalid|expired|not available|does not exist|required/.test(message)) {
    return "invalid_configuration";
  }
  return "unknown";
};

export const redactTelemetry = (value: unknown): unknown => {
  if (
    value === null ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "string") {
    if (SECRET_VALUE.test(value)) return "[REDACTED]";
    return value.slice(0, 120);
  }
  if (Array.isArray(value)) return value.map(redactTelemetry);
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [
        key,
        SENSITIVE_KEY.test(key) ? "[REDACTED]" : redactTelemetry(item),
      ]),
    );
  }
  return String(value).slice(0, 120);
};

export class SetupTelemetry {
  readonly #url: string;
  readonly #fetch: typeof globalThis.fetch;
  readonly #disabled: boolean;
  #telemetryToken: string | undefined;

  constructor(
    apiUrl: string,
    options: {
      fetch?: typeof globalThis.fetch;
      disabled?: boolean;
    } = {},
  ) {
    this.#url = `${apiUrl.replace(/\/+$/, "")}/cli/analytics`;
    this.#fetch = options.fetch ?? globalThis.fetch;
    this.#disabled =
      options.disabled ?? process.env.CONFIDENT_TELEMETRY_DISABLED === "1";
  }

  setTelemetryToken(telemetryToken: string | undefined): void {
    this.#telemetryToken = telemetryToken;
  }

  async send(event: TelemetryEvent): Promise<void> {
    if (this.#disabled || !this.#telemetryToken) return;
    try {
      await this.#fetch(this.#url, {
        method: "POST",
        headers: {
          authorization: `Bearer ${this.#telemetryToken}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(redactTelemetry(event)),
      });
    } catch {
      // Setup telemetry is intentionally best-effort.
    }
  }
}
