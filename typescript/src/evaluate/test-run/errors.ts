import { MetricData } from "@/evaluate/types";

export function buildFailureMessage(metricsData: MetricData[]): string {
  // A score-only metric never failed — it had no verdict to give.
  const failed = metricsData.filter(
    (m) => !m.skipped && (m.error != null || m.success === false),
  );
  const parts = failed.map(
    (m) =>
      `${m.name} (score: ${m.score}, threshold: ${m.threshold}, ` +
      `strict: ${m.strictMode}, error: ${m.error}, reason: ${m.reason}` +
      `${m.flaky ? ", flaky: true" : ""})`,
  );
  return `Metrics: ${parts.join(", ")} failed.`;
}
