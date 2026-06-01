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
