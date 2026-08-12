// A `.d.ts` because the CJS loader and the ESM UI both need these types, and
// anything emitted would collide on one output path. Fields are spelled out
// instead of extending `BaseApiSpan`/`TraceApi` so the UI's compilation stays
// free of the runtime graph; `toInspectSpan` checks them against each other.

type Opaque = unknown;

interface TracedNode {
  uuid?: string;
  name?: string;
  status?: string;
  startTime?: string;
  endTime?: string;
  input?: Opaque;
  output?: Opaque;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: string[];
  toolsCalled?: Opaque[];
  expectedTools?: Opaque[];
  metricsData?: Opaque[];
  metadata?: Record<string, Opaque>;
}

export interface InspectSpan extends TracedNode {
  uuid: string;
  type?: string;
  parentUuid?: string;
  error?: string;

  availableTools?: string[];
  agentHandoffs?: string[];
  description?: string;

  embedder?: string;
  topK?: number;
  chunkSize?: number;

  model?: string;
  inputTokenCount?: number;
  outputTokenCount?: number;
  costPerInputToken?: number;
  costPerOutputToken?: number;

  children: InspectSpan[];
}

export interface InspectTrace extends TracedNode {
  environment?: string;
  tags?: string[];
  threadId?: string;
  userId?: string;
  rootSpans: InspectSpan[];
  caseName?: string;
  casePassed?: boolean;
}

export interface RunSummary {
  testPassed?: number;
  testFailed?: number;
  runDuration?: number;
  evaluationCost?: number;
}

export interface InspectUiOptions {
  traces: InspectTrace[];
  sourcePath: string;
  summary: RunSummary | null;
}

export interface InspectUiModule {
  mount(options: InspectUiOptions): Promise<void>;
}

export interface MetricLike {
  name?: string;
  score?: number | null;
  threshold?: number | null;
  success?: boolean | null;
  reason?: string | null;
  error?: string | null;
  evaluationModel?: string | null;
  evaluationCost?: number | null;
}
