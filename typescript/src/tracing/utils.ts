import { Provider } from "./integrations";

export enum Environment {
  PRODUCTION = "production",
  DEVELOPMENT = "development",
  STAGING = "staging",
  TESTING = "testing",
}

export function tracingEnabled() {
  const envValue =
    process.env.CONFIDENT_TRACING_ENABLED !== undefined
      ? process.env.CONFIDENT_TRACING_ENABLED
      : "YES";
  return envValue.toUpperCase() === "YES";
}

export function validateEnvironment(environment: Environment): void {
  const validEnvironments = Object.values(Environment);
  if (!validEnvironments.includes(environment as Environment)) {
    const validValues = validEnvironments.map((env) => `"${env}"`).join(", ");
    throw new Error(
      `Invalid environment: ${environment}. Please use one of the following instead: ${validValues}`,
    );
  }
}

export function validateSamplingRate(samplingRate: number): void {
  if (samplingRate < 0 || samplingRate > 1) {
    throw new Error(
      `Invalid sampling rate: ${samplingRate}. Please use a value between 0 and 1`,
    );
  }
}

export function toZodCompatibleISO(
  date: Date,
  microsecondPrecision = false,
): string {
  const iso = date.toISOString();

  if (microsecondPrecision) {
    const [base, rest] = iso.split(".");
    const [ms, _z] = rest.split("Z");
    return `${base}.${ms.padEnd(6, "0")}Z`;
  }

  return iso;
}

export function toZodCompatibleIso(dt: Date): string {
  return dt.toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function inferProviderFromModel(model: unknown): string | undefined {
  if (model == null || model === "") {
    return undefined;
  }
  const raw = String(model).trim();
  if (!raw) {
    return undefined;
  }

  const cleanName = raw.toLowerCase().replace(/:/g, "/");
  const modelId = cleanName.split("/").pop() ?? cleanName;

  const prefixMapping: [string, Provider][] = [
    ["gpt", Provider.OPEN_AI],
    ["o1", Provider.OPEN_AI],
    ["o3", Provider.OPEN_AI],
    ["o4", Provider.OPEN_AI],
    ["text-embedding-", Provider.OPEN_AI],
    ["gemini", Provider.GEMINI],
    ["palm", Provider.GEMINI],
    ["gecko", Provider.GEMINI],
    ["claude", Provider.ANTHROPIC],
    ["sonnet", Provider.ANTHROPIC],
    ["opus", Provider.ANTHROPIC],
    ["haiku", Provider.ANTHROPIC],
    ["mistral", Provider.MISTRAL],
    ["mixtral", Provider.MISTRAL],
    ["pixtral", Provider.MISTRAL],
    ["codestral", Provider.MISTRAL],
    ["grok", Provider.X_AI],
    ["deepseek", Provider.DEEP_SEEK],
  ];

  for (const [prefix, provider] of prefixMapping) {
    if (modelId.startsWith(prefix)) {
      return provider;
    }
  }

  const seen = new Set<string>();
  for (const [, provider] of prefixMapping) {
    if (seen.has(provider)) continue;
    seen.add(provider);
    if (cleanName.includes(provider.toLowerCase())) {
      return provider;
    }
  }

  return undefined;
}

function normalizeProviderString(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

export function normalizeSpanProviderForPlatform(
  raw: unknown,
): string | undefined {
  if (raw == null) return undefined;
  const s = String(raw).trim();
  if (!s) return undefined;

  const normalizedRaw = normalizeProviderString(s);
  const head = s.split(/[\s./\\]+/, 1)[0] ?? s;
  const normalizedHead = normalizeProviderString(head);

  for (const [enumKey, canonical] of Object.entries(Provider)) {
    const normalizedCanonical = normalizeProviderString(canonical);
    const normalizedEnumKey = normalizeProviderString(enumKey);
    if (
      normalizedRaw === normalizedCanonical ||
      normalizedRaw === normalizedEnumKey
    ) {
      return canonical;
    }
    if (
      normalizedHead === normalizedCanonical ||
      normalizedHead === normalizedEnumKey
    ) {
      return canonical;
    }
  }

  return s;
}
