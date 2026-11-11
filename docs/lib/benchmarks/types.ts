export type GradingMethod = "programmatic" | "llm-judge" | "mixed";

export type BenchmarkCategory =
  | "ai-agents"
  | "voice-agents"
  | "speech-to-speech"
  | "speech-to-text"
  | "language-models";

export type MetricUnit =
  | "percent"
  | "seconds"
  | "milliseconds"
  | "usd"
  | "multiplier"
  | "count";

export type BenchmarkMetric = {
  id: string;
  label: string;
  // Heading for this metric's card in the "best picks" row. Metrics without
  // one are left out of that row.
  pickLabel?: string;
  unit: MetricUnit;
  higherIsBetter: boolean;
  description: string;
  decimals?: number;
};

export type LeaderboardEntry = {
  model: string;
  provider: string;
  // Keyed by `BenchmarkMetric.id`.
  scores: Record<string, number>;
  // Every metric broken down by segment: `segmentScores[metricId][segmentId]`.
  segmentScores: Record<string, Record<string, number>>;
  // 0 to 100, higher is better, keyed by `BenchmarkSkill.id`.
  skillScores: Record<string, number>;
  // Half-width of the 95% interval on the primary metric, in metric units.
  confidenceInterval?: number;
  openWeights?: boolean;
  evaluatedAt: string;
};

export type RankedEntry = LeaderboardEntry & { rank: number };

export type BenchmarkGrading = {
  method: GradingMethod;
  summary: string;
  judgeModel?: string;
  runsPerCase: number;
};

export type BenchmarkDataset = {
  size: number;
  unit: string;
  source: string;
  license: string;
  host: "Hugging Face" | "GitHub";
  url: string;
  // Share of the test set that is publicly downloadable, 0 to 1.
  publicShare: number;
};

// A slice of the test set that the primary metric is also reported on.
export type BenchmarkSegment = {
  id: string;
  label: string;
  caseCount: number;
};

export type BenchmarkSkill = {
  id: string;
  label: string;
  description: string;
};

export type MethodologyStep = {
  title: string;
  body: string;
};

export type BenchmarkFaq = {
  question: string;
  answer: string;
};

export type Benchmark = {
  slug: string;
  name: string;
  category: BenchmarkCategory;
  tagline: string;
  summary: string;
  version: string;
  updatedAt: string;
  grading: BenchmarkGrading;
  dataset: BenchmarkDataset;
  metrics: BenchmarkMetric[];
  primaryMetricId: string;
  // Plotted against the primary metric in the trade-off chart.
  tradeoffMetricId: string;
  // What the segments divide the test set by, e.g. "Domain".
  segmentLabel: string;
  segments: BenchmarkSegment[];
  skills: BenchmarkSkill[];
  entries: LeaderboardEntry[];
  methodology: MethodologyStep[];
  runSnippet: string;
  faqs: BenchmarkFaq[];
};
