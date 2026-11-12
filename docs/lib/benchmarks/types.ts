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

export type JourneyLine = {
  id: string;
  say: string;
  audio: string;
  // What else happens on this line: a pause, a cough, someone else talking.
  events: { type: string; label: string; audio?: string }[];
  ending?: { name: string; result: "expected" | "failure" };
  // The caller only says this after a serious mistake by the agent.
  failsCall?: boolean;
  maxVisits?: number;
};

export type JourneyReaction = {
  id: string;
  say: string;
  audio: string;
  when: string;
  maxVisits: number;
  then: string;
};

export type JourneyNodeKind =
  | "start"
  | "state"
  | "group"
  | "correction"
  | "end"
  | "failure";

export type JourneyGraphNode = {
  id: string;
  label: string;
  kind: JourneyNodeKind;
  // The `JourneyLine.id`s said at this node. A group has several.
  lineIds: string[];
  // How many things happen on its lines besides the caller speaking.
  events: number;
  x: number;
  y: number;
  // On the shortest route to an expected ending.
  main: boolean;
};

export type JourneyGraphEdge = {
  id: string;
  from: string;
  to: string;
  // What the agent does to send the caller this way.
  when: string;
  points: number[][];
  label: number[];
  main: boolean;
  back: boolean;
};

export type JourneyGraph = {
  start: string;
  nodes: JourneyGraphNode[];
  edges: JourneyGraphEdge[];
  width: number;
  height: number;
};

// A scripted caller as a graph: what they say, and where each agent reply
// sends them next.
export type BenchmarkJourney = {
  id: string;
  // A few words, for the list of journeys.
  name: string;
  // What makes the call hard, in a few words.
  blurb: string;
  title: string;
  // A `BenchmarkSegment.id`.
  segmentId: string;
  summary: string;
  skills: string[];
  caller: string;
  // The line and room the call is made from, e.g. "Speakerphone".
  conditions: string[];
  outcome: string[];
  shape: {
    callerLines: number;
    endings: number;
    branchPoints: number;
    shortestPath: number;
  };
  // Laid out ahead of time by scripts/layout-journeys.mjs.
  graph: JourneyGraph;
  lines: JourneyLine[];
  reactions: JourneyReaction[];
  expectedTools: string[];
  forbiddenTools: string[];
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
  // Only benchmarks whose test cases are caller journeys have these.
  journeys?: BenchmarkJourney[];
};
