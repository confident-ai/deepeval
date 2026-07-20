import { MetricData } from "../types";

export class AssertionFailedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AssertionFailedError";
  }
}

export function buildFailureMessage(metricsData: MetricData[]): string {
  const failed = metricsData.filter(
    (m) => !m.skipped && (m.error != null || !m.success),
  );
  const parts = failed.map(
    (m) =>
      `${m.name} (score: ${m.score}, threshold: ${m.threshold}, ` +
      `strict: ${m.strictMode}, error: ${m.error}, reason: ${m.reason})`,
  );
  return `Metrics: ${parts.join(", ")} failed.`;
}
