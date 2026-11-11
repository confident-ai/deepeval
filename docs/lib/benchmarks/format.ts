import type {
  Benchmark,
  BenchmarkCategory,
  BenchmarkMetric,
  GradingMethod,
  LeaderboardEntry,
  RankedEntry,
} from "./types";

export const CATEGORY_LABELS: Record<BenchmarkCategory, string> = {
  "ai-agents": "AI agents",
  "voice-agents": "Voice agents",
  "speech-to-speech": "Speech-to-speech",
  "speech-to-text": "Speech-to-text",
  "language-models": "Language models",
};

export const GRADING_LABELS: Record<GradingMethod, string> = {
  programmatic: "Programmatic",
  "llm-judge": "LLM judge",
  mixed: "Programmatic + LLM judge",
};

export function formatMetric(value: number, metric: BenchmarkMetric): string {
  const decimals = metric.decimals ?? 1;
  const fixed = value.toFixed(decimals);
  switch (metric.unit) {
    case "percent":
      return `${fixed}%`;
    case "seconds":
      return `${fixed}s`;
    case "milliseconds":
      return `${fixed}ms`;
    case "usd":
      return `$${fixed}`;
    case "multiplier":
      return `${fixed}×`;
    case "count":
      return fixed;
  }
}

// Ranks are always shown as two digits: 01, 02, 10.
export function formatRank(rank: number): string {
  return String(rank).padStart(2, "0");
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatDate(iso: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(new Date(iso));
}

export function getMetric(
  metrics: BenchmarkMetric[],
  id: string,
): BenchmarkMetric {
  const metric = metrics.find((item) => item.id === id);
  if (!metric) throw new Error(`Unknown metric: ${id}`);
  return metric;
}

export function sortEntries(
  entries: LeaderboardEntry[],
  metric: BenchmarkMetric,
  direction: "best-first" | "worst-first" = "best-first",
): LeaderboardEntry[] {
  const sign = metric.higherIsBetter ? -1 : 1;
  const flip = direction === "best-first" ? 1 : -1;
  return [...entries].sort(
    (a, b) => sign * flip * (a.scores[metric.id] - b.scores[metric.id]),
  );
}

// Rank is always by the benchmark's primary metric, whatever the table is
// currently sorted by.
export function rankEntries(benchmark: Benchmark): RankedEntry[] {
  const primary = getMetric(benchmark.metrics, benchmark.primaryMetricId);
  return sortEntries(benchmark.entries, primary).map((entry, index) => ({
    ...entry,
    rank: index + 1,
  }));
}

// Bar length for a score, 0 to 1, relative to the best value in the column.
export function scoreFraction(
  value: number,
  metric: BenchmarkMetric,
  values: number[],
): number {
  if (values.length === 0) return 0;
  if (metric.higherIsBetter) {
    const max = Math.max(...values);
    return max > 0 ? value / max : 0;
  }
  const min = Math.min(...values);
  return value > 0 ? min / value : 0;
}

export function providerInitials(provider: string): string {
  const words = provider.split(/\s+/).filter(Boolean);
  if (words.length > 1) return (words[0][0] + words[1][0]).toUpperCase();
  return provider.slice(0, 2);
}

export type TradeoffPoint = {
  model: string;
  provider: string;
  rank?: number;
  x: number;
  y: number;
};

// Models that no other model beats on both axes at once.
export function paretoFront(
  points: TradeoffPoint[],
  xHigherIsBetter: boolean,
  yHigherIsBetter: boolean,
): TradeoffPoint[] {
  const better = (a: number, b: number, higher: boolean) =>
    higher ? a >= b : a <= b;
  const strictly = (a: number, b: number, higher: boolean) =>
    higher ? a > b : a < b;

  return points
    .filter(
      (point) =>
        !points.some(
          (other) =>
            other !== point &&
            better(other.x, point.x, xHigherIsBetter) &&
            better(other.y, point.y, yHigherIsBetter) &&
            (strictly(other.x, point.x, xHigherIsBetter) ||
              strictly(other.y, point.y, yHigherIsBetter)),
        ),
    )
    .sort((a, b) => a.x - b.x);
}
