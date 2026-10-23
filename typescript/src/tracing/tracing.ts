import { AsyncLocalStorage } from "async_hooks";
import console from "console";
import {
  context as otelContext,
  type Span as OtelSpan,
  SpanStatusCode,
} from "@opentelemetry/api";

import { getSettings } from "@/config/settings";
import { getConfidentApiKey } from "@/utils";
import {
  validateEnvironment,
  validateSamplingRate,
  toZodCompatibleISO,
  Environment,
} from "@/tracing/utils";

import { LLMTestCase, ToolCall, resolveRetrievalContext } from "@/test-case";
import type { BaseMetric } from "@/metrics/base-metrics";
import { type MetricData, withoutExpectations } from "@/evaluate/types";
import { Prompt } from "@/prompt";
import {
  SpanApiType,
  BaseApiSpan,
  TraceApi,
  TraceSpanApiStatus,
} from "@/tracing/api";
import {
  type PendingPayload,
  slotKindForSpanType,
} from "@/tracing/pending-context";
import { ConfidentAttr } from "@/tracing/attributes";
import { isTraceInternalEnabled } from "@/tracing/utils";
import { otelHooks } from "@/tracing/otel/hooks";

export enum SpanType {
  AGENT = "agent",
  LLM = "llm",
  RETRIEVER = "retriever",
  TOOL = "tool",
  CUSTOM = "custom",
}

export enum TraceSpanStatus {
  SUCCESS = "SUCCESS",
  ERRORED = "ERRORED",
  IN_PROGRESS = "IN_PROGRESS",
}

////////////////////////////////////////////////////////
/// Attributes /////////////////////////////////////////
////////////////////////////////////////////////////////

export interface ToolAttributes {
  inputParameters?: Record<string, any>;
  output?: any;
}

////////////////////////////////////////////////////////
/// Trace Types ///////////////////////////////////////
////////////////////////////////////////////////////////

// Base class
export class BaseSpan {
  uuid: string;
  status: TraceSpanStatus;
  children: BaseSpan[] = [];
  traceUuid: string;
  parentUuid?: string;
  startTime: Date;
  endTime?: Date;
  name?: string;
  input?: any;
  output?: any;
  error?: string;
  metricCollection?: string;
  metrics?: BaseMetric[];
  /** Metric results computed locally by the trace-eval executor (for posting). */
  metricsData?: MetricData[];
  type: SpanType | string;
  metadata?: Record<string, any>;
  retrievalContext?: string[];
  context?: string[];
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  expectedOutput?: string;

  constructor(params: {
    uuid: string;
    status: TraceSpanStatus;
    traceUuid: string;
    startTime: Date;
    type: SpanType | string;
    parentUuid?: string;
    name?: string;
    input?: any;
    output?: any;
    error?: string;
    metricCollection?: string;
    metrics?: BaseMetric[];
    metadata?: Record<string, any>;
    retrievalContext?: string[];
    context?: string[];
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    expectedOutput?: string;
  }) {
    this.uuid = params.uuid;
    this.status = params.status;
    this.traceUuid = params.traceUuid;
    this.startTime = params.startTime;
    this.type = params.type;
    this.parentUuid = params.parentUuid;
    this.name = params.name;
    this.input = params.input;
    this.output = params.output;
    this.error = params.error;
    this.metricCollection = params.metricCollection;
    this.metrics = params.metrics;
    this.metadata = params.metadata;
    this.retrievalContext = params.retrievalContext;
    this.context = params.context;
    this.toolsCalled = params.toolsCalled;
    this.expectedTools = params.expectedTools;
    this.expectedOutput = params.expectedOutput;
  }
}

// AgentSpan
export class AgentSpan extends BaseSpan {
  name: string;
  availableTools: string[];
  agentHandoffs: string[];

  constructor(params: {
    uuid: string;
    status: TraceSpanStatus;
    traceUuid: string;
    startTime: Date;
    type: SpanType | string;
    availableTools: string[];
    agentHandoffs: string[];
    parentUuid?: string;
    name: string;
    input?: any;
    output?: any;
    error?: string;
    metricCollection?: string;
    metadata?: Record<string, any>;
    retrievalContext?: string[];
    context?: string[];
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    expectedOutput?: string;
  }) {
    super(params);
    this.name = params.name;
    this.availableTools = params.availableTools;
    this.agentHandoffs = params.agentHandoffs;
  }
}

// LlmSpan
export class LlmSpan extends BaseSpan {
  model: string;
  prompt?: Prompt;
  promptCommitHash?: string;
  promptAlias?: string;
  promptLabel?: string;
  promptVersion?: string;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  inputTokenCount?: number;
  outputTokenCount?: number;
  tokenInterval?: number;

  constructor(params: {
    uuid: string;
    status: TraceSpanStatus;
    traceUuid: string;
    startTime: Date;
    type: SpanType | string;
    model: string;
    costPerInputToken?: number;
    costPerOutputToken?: number;
    inputTokenCount?: number;
    outputTokenCount?: number;
    parentUuid?: string;
    name?: string;
    input?: any;
    output?: any;
    error?: string;
    metricCollection?: string;
    metadata?: Record<string, any>;
    retrievalContext?: string[];
    context?: string[];
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    expectedOutput?: string;
  }) {
    super(params);
    this.model = params.model;
    this.costPerInputToken = params.costPerInputToken;
    this.costPerOutputToken = params.costPerOutputToken;
    this.inputTokenCount = params.inputTokenCount;
    this.outputTokenCount = params.outputTokenCount;
  }
}

// RetrieverSpan
export class RetrieverSpan extends BaseSpan {
  embedder: string;
  topK?: number;
  chunkSize?: number;

  constructor(params: {
    uuid: string;
    status: TraceSpanStatus;
    traceUuid: string;
    startTime: Date;
    type: SpanType | string;
    embedder: string;
    topK?: number;
    chunkSize?: number;
    parentUuid?: string;
    name?: string;
    input?: any;
    output?: any;
    error?: string;
    metricCollection?: string;
    metadata?: Record<string, any>;
    retrievalContext?: string[];
    context?: string[];
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    expectedOutput?: string;
  }) {
    super(params);
    this.embedder = params.embedder;
    this.topK = params.topK;
    this.chunkSize = params.chunkSize;
  }
}

// ToolSpan
export class ToolSpan extends BaseSpan {
  name: string; // required
  description?: string;

  constructor(params: {
    uuid: string;
    status: TraceSpanStatus;
    traceUuid: string;
    startTime: Date;
    type: SpanType | string;
    name: string;
    description?: string;
    parentUuid?: string;
    input?: any;
    output?: any;
    error?: string;
    metricCollection?: string;
    metadata?: Record<string, any>;
    retrievalContext?: string[];
    context?: string[];
    toolsCalled?: ToolCall[];
    expectedTools?: ToolCall[];
    expectedOutput?: string;
  }) {
    super(params);
    this.name = params.name;
    this.description = params.description;
  }
}

export interface Trace {
  uuid: string;
  status: TraceSpanStatus;
  rootSpans: BaseSpan[];
  startTime: Date;
  endTime?: Date;
  metadata?: Record<string, any>;
  tags?: string[];
  threadId?: string;
  userId?: string;
  testCaseId?: string;
  testRunId?: string;
  turnId?: string;
  input?: any;
  output?: any;
  name?: string;
  expectedOutput?: string;
  retrievalContext?: string[];
  context?: string[];
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  metricCollection?: string;
  metrics?: BaseMetric[];
  metricsData?: MetricData[];
  confidentApiKey?: string;
  drop?: boolean;
}

////////////////////////////////////////////////////////
/// Tracing Context /////////////////////////////////////
////////////////////////////////////////////////////////

const tracingContext = new AsyncLocalStorage<{
  currentSpan?: BaseSpan;
  currentTrace?: Trace;
}>();

export function getCurrentSpan(): BaseSpan | undefined {
  return tracingContext.getStore()?.currentSpan;
}

export function getCurrentTrace(): Trace | undefined {
  return tracingContext.getStore()?.currentTrace;
}

export function setCurrentSpan(span: BaseSpan | null): void {
  const current = tracingContext.getStore() ?? {};
  tracingContext.enterWith({
    ...current,
    currentSpan: span ?? undefined,
  });
}

export function setCurrentTrace(trace: Trace | null): void {
  const current = tracingContext.getStore() ?? {};
  tracingContext.enterWith({
    ...current,
    currentTrace: trace ?? undefined,
  });
}

/**
 * The innermost open span: deepeval's own, or a confident-trace span deepeval
 * collects, seen through the placeholder that stands in for it.
 */
export function activeSpan(): BaseSpan | undefined {
  const span = getCurrentSpan();
  const collected = otelHooks.collectedSpan?.();
  if (!collected || (span && isDescendant(span, collected.uuid))) return span;
  return collected;
}

/** The trace of `activeSpan()`. */
export function activeTrace(): Trace | undefined {
  return getCurrentTrace() ?? otelHooks.collectedTrace?.();
}

/** Whether `span` was opened inside the span `ancestorUuid`. */
export function isDescendant(span: BaseSpan, ancestorUuid: string): boolean {
  let current: BaseSpan | undefined = span;
  while (current?.parentUuid) {
    if (current.parentUuid === ancestorUuid) return true;
    current = traceManager.getSpanByUuid(current.parentUuid);
  }
  return false;
}

export function withTracingContext<T>(
  span: BaseSpan | undefined,
  trace: Trace | undefined,
  callback: () => T,
): T {
  const parent = tracingContext.getStore() ?? {};
  const context = {
    currentSpan: span ?? parent.currentSpan,
    currentTrace: trace ?? parent.currentTrace,
  };
  return tracingContext.run(context, callback);
}

////////////////////////////////////////////////////////
/// TraceManager ///////////////////////////////////////
////////////////////////////////////////////////////////

export type MaskFunction = (data: any) => any;

export interface TraceManagerConfig {
  environment?: Environment;
  samplingRate?: number;
  mask?: MaskFunction;
  confidentApiKey?: string;
  tracingEnabled?: boolean;
}

export class TraceManager {
  private traces: Trace[] = [];
  private activeTraces: Map<string, Trace> = new Map(); // Map of traceUuid to Trace
  private activeSpans: Map<string, BaseSpan> = new Map(); // Map of spanUuid to BaseSpan

  // Configuration
  private environment: Environment;
  private samplingRate: number;
  private customMaskFn: MaskFunction | null = null;
  private evaluatingDepth: number = 0;
  private traceCaptureSinks = new Set<(trace: Trace) => void>();
  private settleHooks = new Set<() => Promise<void>>();
  private confidentApiKey: string = "";
  private tracingEnabled: boolean = true;

  public addTraceCaptureSink(sink: (trace: Trace) => void): () => void {
    this.traceCaptureSinks.add(sink);
    return () => {
      this.traceCaptureSinks.delete(sink);
    };
  }

  public setTraceCaptureSink(sink?: (trace: Trace) => void): void {
    this.traceCaptureSinks.clear();
    if (sink) this.traceCaptureSinks.add(sink);
  }

  public addSettleHook(hook: () => Promise<void>): () => void {
    this.settleHooks.add(hook);
    return () => {
      this.settleHooks.delete(hook);
    };
  }

  public get isEvaluating(): boolean {
    return this.evaluatingDepth > 0;
  }

  /** Whether traces are consumed locally: by an evaluation or a capture sink. */
  public get isCapturingTraces(): boolean {
    return this.isEvaluating || this.traceCaptureSinks.size > 0;
  }

  public beginEvaluation(): () => void {
    this.evaluatingDepth += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.evaluatingDepth = Math.max(0, this.evaluatingDepth - 1);
    };
  }

  /** Await every registered settle hook. A failing hook must not fail the run. */
  public async awaitSettled(): Promise<void> {
    if (this.settleHooks.size === 0) return;
    await Promise.all(
      Array.from(this.settleHooks).map((hook) =>
        hook().catch((error) => {
          if (this.settings.CONFIDENT_TRACE_VERBOSE) {
            console.warn("[DeepEval] settle hook failed:", error);
          }
        }),
      ),
    );
  }

  private settings = getSettings();

  constructor() {
    this.environment =
      this.settings.CONFIDENT_TRACE_ENVIRONMENT !== undefined &&
      this.settings.CONFIDENT_TRACE_ENVIRONMENT !== null
        ? this.settings.CONFIDENT_TRACE_ENVIRONMENT
        : Environment.DEVELOPMENT;
    validateEnvironment(this.environment);
    this.samplingRate = parseFloat(process.env.CONFIDENT_SAMPLE_RATE || "1");
    validateSamplingRate(this.samplingRate);
  }

  public mask(data: any): any {
    if (this.customMaskFn) {
      return this.customMaskFn(data);
    } else {
      return data;
    }
  }

  /**
   * Serialize a span subtree into a nested dict for trace-based metrics
   * (mirrors Python's `create_nested_spans_dict`). Keeps the eval-relevant
   * fields + recurses children; drops bookkeeping (uuids, times, status,
   * metrics, metricCollection, metadata) and null values.
   */
  public createNestedSpansDict(span: BaseSpan): Record<string, unknown> {
    const dict: Record<string, unknown> = {};
    const put = (k: string, v: unknown) => {
      if (v !== undefined && v !== null) dict[k] = v;
    };
    put("name", span.name);
    put("type", span.type);
    put("input", span.input);
    put("output", span.output);
    put("error", span.error);
    put("expectedOutput", span.expectedOutput);
    put("context", span.context);
    put("retrievalContext", span.retrievalContext);
    put("toolsCalled", span.toolsCalled);
    put("expectedTools", span.expectedTools);
    // Span-type extras (model / tools) when present, like Python's api span.
    put("model", (span as { model?: unknown }).model);
    put(
      "availableTools",
      (span as { availableTools?: unknown }).availableTools,
    );
    dict.children = (span.children ?? []).map((child) =>
      this.createNestedSpansDict(child),
    );
    return dict;
  }

  public configure(config: TraceManagerConfig): void {
    if (config.environment) {
      validateEnvironment(config.environment);
      this.environment = config.environment;
    }
    if (config.samplingRate !== undefined) {
      validateSamplingRate(config.samplingRate);
      this.samplingRate = config.samplingRate;
    }
    if (config.mask !== undefined) {
      this.customMaskFn = config.mask;
    }
    if (config.confidentApiKey !== undefined) {
      this.confidentApiKey = config.confidentApiKey;
    }
    if (config.tracingEnabled !== undefined) {
      this.tracingEnabled = config.tracingEnabled;
    }
  }

  public startNewTrace(traceUuid: string = crypto.randomUUID()): Trace {
    const newTrace: Trace = {
      uuid: traceUuid,
      rootSpans: [],
      status: TraceSpanStatus.IN_PROGRESS,
      startTime: new Date(),
      endTime: undefined,
    };
    this.activeTraces.set(traceUuid, newTrace);
    this.traces.push(newTrace);
    return newTrace;
  }

  public endTrace(traceUuid: string): void {
    const trace = this.activeTraces.get(traceUuid);
    if (trace) {
      trace.endTime = new Date();
      if (trace.status === TraceSpanStatus.IN_PROGRESS) {
        trace.status = TraceSpanStatus.SUCCESS;
      }
      // Otherwise confident-trace exports the trace's spans.
      for (const sink of this.traceCaptureSinks) sink(trace);
      this.activeTraces.delete(traceUuid);
    }
  }

  /** Drop an unfinished trace without capturing or posting it. */
  public discardTrace(traceUuid: string): void {
    const trace = this.activeTraces.get(traceUuid);
    if (!trace) return;
    this.activeTraces.delete(traceUuid);
    this.traces = this.traces.filter((t) => t !== trace);
  }

  /** The Confident AI key traces are sent with, if any. */
  public get exportApiKey(): string | undefined {
    return this.confidentApiKey || getConfidentApiKey() || undefined;
  }

  /** The share of traces exported to Confident AI. */
  public get exportSamplingRate(): number {
    return this.samplingRate;
  }

  /** The environment an exported trace carries unless it sets one. */
  public get exportEnvironment(): string {
    return this.environment;
  }

  /** Whether `configure({ tracingEnabled })` lets traces export. */
  public get exportEnabled(): boolean {
    return this.tracingEnabled;
  }

  public setTraceStatus(traceUuid: string, status: TraceSpanStatus): void {
    const trace = this.activeTraces.get(traceUuid);
    if (trace) {
      trace.status = status;
    }
  }

  public addSpan(span: BaseSpan): void {
    this.activeSpans.set(span.uuid, span);
  }

  public removeSpan(spanUuid: string): void {
    this.activeSpans.delete(spanUuid);
  }

  public addSpanToTrace(span: BaseSpan): void {
    const traceUuid = span.traceUuid;
    const trace = this.activeTraces.get(traceUuid);
    if (!trace) {
      throw new Error(
        `Trace with UUID ${traceUuid} does not exist. A span must have a valid trace.`,
      );
    }
    // If this is a root span (no parent), add it to the trace's rootSpans
    if (!span.parentUuid) {
      trace.rootSpans.push(span);
    } else {
      // This is a child span, find its parent and add it to the parent's children
      const parentSpan = this.getSpanByUuid(span.parentUuid);
      if (parentSpan) {
        parentSpan.children.push(span);
      } else {
        throw new Error(
          `Parent span with UUID ${span.parentUuid} does not exist.`,
        );
      }
    }
  }

  // Update the span in its trace
  public getTraceByUuid(traceUuid: string): Trace | undefined {
    return this.activeTraces.get(traceUuid);
  }

  public getSpanByUuid(spanUuid: string): BaseSpan | undefined {
    return this.activeSpans.get(spanUuid);
  }

  public getAllTraces(): Trace[] {
    return this.traces;
  }

  public clearTraces(): void {
    this.traces = [];
    this.activeTraces.clear();
    this.activeSpans.clear();
    // Reset async-local trace/span pointers so callers (notably test
    // `beforeEach`) get a fresh context. Without this, AsyncLocalStorage
    // leaks a stale trace UUID across tests and span attachment throws.
    setCurrentTrace(null);
    setCurrentSpan(null);
  }

  /** Wait up to `timeoutMillis` for confident-trace to export every ended span. */
  public async flush(timeoutMillis = 30000): Promise<boolean> {
    return (await otelHooks.flush?.(timeoutMillis)) ?? true;
  }

  public createTraceApi(trace: Trace): TraceApi {
    // Initialize empty arrays for each span type
    const baseSpans: BaseApiSpan[] = [];
    const agentSpans: BaseApiSpan[] = [];
    const llmSpans: BaseApiSpan[] = [];
    const retrieverSpans: BaseApiSpan[] = [];
    const toolSpans: BaseApiSpan[] = [];
    // Process all spans in the trace, including child spans
    const processSpans = (spans: BaseSpan[]) => {
      for (const span of spans) {
        // Convert BaseSpan to BaseApiSpan
        const apiSpan = this.convertSpanToApiSpan(span);
        // Ensure required fields are present based on span type
        if (apiSpan.type === SpanType.RETRIEVER && !apiSpan.embedder) {
          apiSpan.embedder = "default-embedder";
        }
        // Categorize spans by type and add ALL spans (not just root spans)
        if (apiSpan.type === SpanType.AGENT) {
          agentSpans.push(apiSpan);
        } else if (apiSpan.type === SpanType.LLM) {
          llmSpans.push(apiSpan);
        } else if (apiSpan.type === SpanType.RETRIEVER) {
          retrieverSpans.push(apiSpan);
        } else if (apiSpan.type === SpanType.TOOL) {
          toolSpans.push(apiSpan);
        } else {
          baseSpans.push(apiSpan);
        }
        // Process children recursively
        if (span.children && span.children.length > 0) {
          processSpans(span.children);
        }
      }
    };
    // Start processing from root spans
    processSpans(trace.rootSpans);
    // Ensure proper datetime formatting for start and end times
    const startTime = trace.startTime
      ? toZodCompatibleISO(trace.startTime)
      : toZodCompatibleISO(new Date());
    // If no end time or if end time is before start time, use start time
    let endTime;
    if (!trace.endTime || trace.endTime < trace.startTime) {
      endTime = startTime;
    } else {
      endTime = toZodCompatibleISO(trace.endTime);
    }
    // Create and return the TraceApi object
    return {
      name: trace.name,
      uuid: trace.uuid,
      baseSpans,
      agentSpans,
      llmSpans,
      retrieverSpans,
      toolSpans,
      startTime,
      endTime,
      environment: this.environment,
      metadata: trace.metadata,
      tags: trace.tags,
      threadId: trace.threadId,
      userId: trace.userId,
      testCaseId: trace.testCaseId,
      testRunId: trace.testRunId,
      turnId: trace.turnId,
      input: trace.input,
      output: trace.output,
      toolsCalled: trace.toolsCalled,
      metricCollection: trace.metricCollection,
      metricsData: trace.metricsData && withoutExpectations(trace.metricsData),
      confidentApiKey: trace.confidentApiKey,
      status:
        trace.status === TraceSpanStatus.SUCCESS
          ? TraceSpanApiStatus.SUCCESS
          : TraceSpanApiStatus.ERROR,
    };
  }

  // Convert a BaseSpan to a BaseApiSpan
  private convertSpanToApiSpan(span: BaseSpan): BaseApiSpan {
    let _inputData: any = span.input;
    let _outputData: any = span.output;

    // Ensure input and output are not undefined
    _inputData = _inputData ?? "";
    _outputData = _outputData ?? (span.type === SpanType.RETRIEVER ? [] : "");

    // Convert Date objects to ISO strings
    const startTime = span.startTime
      ? toZodCompatibleISO(span.startTime)
      : toZodCompatibleISO(new Date());
    // If no end time or if end time is before start time, use start time
    let endTime;
    if (!span.endTime || span.endTime < span.startTime) {
      endTime = startTime;
    } else {
      endTime = toZodCompatibleISO(span.endTime);
    }
    // Create the base API span
    const apiSpan: BaseApiSpan = {
      uuid: span.uuid,
      name: span.name,
      status:
        span.status === TraceSpanStatus.SUCCESS
          ? TraceSpanApiStatus.SUCCESS
          : TraceSpanApiStatus.ERROR,
      type: span.type as SpanApiType,
      traceUuid: span.traceUuid,
      parentUuid: span.parentUuid || undefined,
      startTime,
      endTime,
      input: span.input,
      output: span.output,
      error: span.error,
      metricCollection: span.metricCollection,
      metricsData: span.metricsData,
      expectedOutput: span.expectedOutput,
      retrievalContext: span.retrievalContext,
      context: span.context,
      toolsCalled: span.toolsCalled,
      expectedTools: span.expectedTools,
    };

    // Add type-specific fields
    if (span.type === SpanType.AGENT) {
      const agentSpan = span as AgentSpan;
      apiSpan.availableTools = agentSpan.availableTools || [];
      apiSpan.agentHandoffs = agentSpan.agentHandoffs || [];
    } else if (span.type === SpanType.LLM) {
      const llmSpan = span as LlmSpan;
      apiSpan.model = llmSpan.model;
      apiSpan.costPerInputToken = llmSpan.costPerInputToken;
      apiSpan.costPerOutputToken = llmSpan.costPerOutputToken;
      apiSpan.inputTokenCount = llmSpan.inputTokenCount;
      apiSpan.outputTokenCount = llmSpan.outputTokenCount;
      apiSpan.promptAlias = llmSpan.prompt?._alias || llmSpan.promptAlias;
      apiSpan.promptCommitHash =
        llmSpan.prompt?.hash || llmSpan.promptCommitHash;
      apiSpan.promptLabel = llmSpan.prompt?.label || llmSpan.promptLabel;
      apiSpan.promptVersion = llmSpan.prompt?.version || llmSpan.promptVersion;
    } else if (span.type === SpanType.RETRIEVER) {
      const retrieverSpan = span as RetrieverSpan;
      apiSpan.embedder = retrieverSpan.embedder || "default-embedder";
      apiSpan.topK = retrieverSpan.topK;
      apiSpan.chunkSize = retrieverSpan.chunkSize;
    } else if (span.type === SpanType.TOOL) {
      const toolSpan = span as ToolSpan;
      apiSpan.description = toolSpan.description;
    } else {
      span.type = "base";
      apiSpan.type = span.type;
    }
    // Add test case information if available
    if (span.expectedOutput) {
      apiSpan.expectedOutput = span.expectedOutput;
    }
    if (span.retrievalContext) {
      apiSpan.retrievalContext = span.retrievalContext;
    }
    if (span.context) {
      apiSpan.context = span.context;
    }
    if (span.toolsCalled) {
      apiSpan.toolsCalled = span.toolsCalled;
    }
    if (span.expectedTools) {
      apiSpan.expectedTools = span.expectedTools;
    }
    // Add metric collection if available
    if (span.metricCollection) {
      apiSpan.metricCollection = span.metricCollection;
    }
    // Add metadata if available
    if (span.metadata) {
      apiSpan.metadata = span.metadata;
    }
    return apiSpan;
  }

  public getActiveSpans(): Map<string, BaseSpan> {
    return this.activeSpans;
  }
}

// Create a singleton instance of the TraceManager
export const traceManager = new TraceManager();

////////////////////////////////////////////////////////
/// Tracer /////////////////////////////////////////////
////////////////////////////////////////////////////////

// The span types confident-trace knows; any other `observe` type is custom.
const CONFIDENT_TRACE_TYPES = new Set<string>([
  SpanType.AGENT,
  SpanType.LLM,
  SpanType.RETRIEVER,
  SpanType.TOOL,
]);

// `observe` options each span type takes.
const OBSERVE_FIELDS: Record<string, string[]> = {
  [SpanType.AGENT]: ["availableTools", "agentHandoffs"],
  [SpanType.LLM]: ["model", "costPerInputToken", "costPerOutputToken"],
  [SpanType.RETRIEVER]: ["embedder"],
  [SpanType.TOOL]: ["description"],
};

function otelSetup(): typeof import("./otel/setup") {
  // Loaded on first use, since setup's modules need this one initialized.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require("./otel/setup");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * One `observe` call: a confident-trace span that deepeval collects.
 *
 * While it runs, deepeval code sees it through the bridge's placeholder, which
 * is the deepeval span the trace gets when it ends.
 */
export class Tracer {
  private readonly spanType: string;
  private readonly name: string;
  private readonly metricCollection?: string;
  private readonly metrics?: BaseMetric[];
  private readonly observeKwargs: Record<string, any>;
  private readonly functionKwargs: Record<string, any>;

  constructor(
    spanType: SpanType | string | null,
    funcName: string,
    metricCollection?: string,
    kwargs: Record<string, any> = {},
    metrics?: BaseMetric[],
  ) {
    this.observeKwargs = kwargs.observeKwargs || {};
    this.functionKwargs = kwargs.functionKwargs || {};

    // Initialize name from options or function name
    this.name = this.observeKwargs.name || funcName;
    this.metricCollection = metricCollection;
    this.metrics = metrics;

    // Initialize span type from parameter or name
    this.spanType = spanType === null ? this.name : spanType;
  }

  /** Run `callback` inside the span. */
  public async trace<T>(callback: () => T | Promise<T>): Promise<T> {
    const setup = otelSetup();
    const parentSpan = activeSpan();
    let trace = activeTrace();
    let startedTrace: string | undefined;
    // A failed async operation may leave an ended trace in context.
    if (!trace || !traceManager.getTraceByUuid(trace.uuid)) {
      trace = traceManager.startNewTrace();
      startedTrace = trace.uuid;
    }
    const values = this.observeValues();
    const started = withTracingContext(parentSpan, trace, () =>
      setup.startObservedSpan(
        this.name,
        CONFIDENT_TRACE_TYPES.has(this.spanType)
          ? this.spanType
          : SpanType.CUSTOM,
        { span: values, spanKind: slotKindForSpanType(this.spanType) },
      ),
    );
    if (!started?.placeholder && startedTrace)
      // Not collected: confident-trace isn't running, or exports the span
      // on its own under a span it exports.
      traceManager.discardTrace(startedTrace);
    if (!started) return await callback();

    const { span, placeholder } = started;
    setup.exportOn(span, "span", values);
    const spanTrace = placeholder
      ? traceManager.getTraceByUuid(placeholder.traceUuid)
      : undefined;
    let result: T | undefined;
    let failed = false;
    let error: unknown;
    try {
      result = await otelContext.with(setup.observedContext(span), () =>
        withTracingContext(
          placeholder,
          spanTrace,
          async () => await callback(),
        ),
      );
      return result;
    } catch (caught) {
      failed = true;
      error = caught;
      throw caught;
    } finally {
      this.finish(span, placeholder, result, failed, error);
    }
  }

  private finish(
    span: OtelSpan,
    placeholder: BaseSpan | undefined,
    result: unknown,
    failed: boolean,
    error: unknown,
  ): void {
    const setup = otelSetup();
    const input = traceManager.mask(this.functionKwargs);
    const output = traceManager.mask(result);
    if (failed)
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: errorMessage(error),
      });
    if (placeholder) {
      placeholder.status = failed
        ? TraceSpanStatus.ERRORED
        : TraceSpanStatus.SUCCESS;
      if (failed) placeholder.error = errorMessage(error);
      placeholder.input =
        placeholder.input === undefined
          ? input
          : traceManager.mask(placeholder.input);
      placeholder.output =
        placeholder.output === undefined
          ? output
          : traceManager.mask(placeholder.output);
    }
    // The content the span exports, masked like deepeval's own.
    setup.exportOn(
      span,
      "span",
      placeholder
        ? { input: placeholder.input, output: placeholder.output }
        : {
            input: setup.carries(span, ConfidentAttr.SPAN_INPUT)
              ? undefined
              : input,
            output: setup.carries(span, ConfidentAttr.SPAN_OUTPUT)
              ? undefined
              : output,
          },
    );
    if (placeholder && placeholder.parentUuid === undefined) {
      setup.exportOn(span, "trace", {
        input: setup.carries(span, ConfidentAttr.TRACE_INPUT)
          ? undefined
          : input,
        output: setup.carries(span, ConfidentAttr.TRACE_OUTPUT)
          ? undefined
          : output,
      });
      const trace = traceManager.getTraceByUuid(placeholder.traceUuid);
      if (trace) {
        if (trace.input === undefined) trace.input = input;
        if (trace.output === undefined) trace.output = output;
        if (failed) trace.status = TraceSpanStatus.ERRORED;
      }
    }
    // The bridge collects the span, and ends the trace with its last span.
    span.end();
  }

  /** The span fields `observe` was given for this span. */
  private observeValues(): PendingPayload {
    const values: PendingPayload = {
      metrics: this.metrics,
      metricCollection: this.metricCollection,
    };
    for (const key of OBSERVE_FIELDS[this.spanType] ?? [])
      values[key] = this.observeKwargs[key];
    return values;
  }
}

interface AgentOptions<Args extends any[], T> {
  availableTools?: string[] | undefined;
  name?: string | undefined;
  agentHandoffs?: string[] | undefined;
  metricCollection?: string | undefined;
  metrics?: BaseMetric[];
  fn: (...args: Args) => T | Promise<T>;
}

interface LLMOptions<Args extends any[], T> {
  model?: string;
  costPerInputToken?: number | undefined;
  costPerOutputToken?: number | undefined;
  name?: string | undefined;
  metricCollection?: string | undefined;
  metrics?: BaseMetric[];
  fn: (...args: Args) => T | Promise<T>;
}

interface RetrieverOptions<Args extends any[], T> {
  embedder?: string;
  name?: string | undefined;
  metricCollection?: string | undefined;
  metrics?: BaseMetric[];
  fn: (...args: Args) => T | Promise<T>;
}

interface ToolOptions<Args extends any[], T> {
  description?: string | undefined;
  name?: string | undefined;
  metricCollection?: string | undefined;
  metrics?: BaseMetric[];
  fn: (...args: Args) => T | Promise<T>;
}

interface CustomOptions<Args extends any[], T> {
  type: string;
  name?: string | undefined;
  metricCollection?: string | undefined;
  metrics?: BaseMetric[];
  fn: (...args: Args) => T | Promise<T>;
}

function ObserveAgent<Args extends any[], T>(
  options: AgentOptions<Args, T>,
): (...args: Args) => Promise<T> {
  return async (...args: Args): Promise<T> => {
    const defaultName = options.name || options.fn.name || "Unknown";
    const tracer = new Tracer(
      SpanType.AGENT,
      defaultName,
      options.metricCollection,
      {
        observeKwargs: {
          name: defaultName,
          availableTools: options.availableTools,
          agentHandoffs: options.agentHandoffs,
        },
        functionKwargs: { args },
      },
      options.metrics,
    );

    return tracer.trace(async () => {
      const result = options.fn(...args);
      const finalResult = result instanceof Promise ? await result : result;
      return finalResult;
    });
  };
}

function ObserveLLM<Args extends any[], T>(
  options: LLMOptions<Args, T>,
): (...args: Args) => Promise<T> {
  return async (...args: Args): Promise<T> => {
    const defaultName = options.name || options.fn.name || "Unknown";
    const tracer = new Tracer(
      SpanType.LLM,
      defaultName,
      options.metricCollection,
      {
        observeKwargs: {
          name: options.name,
          model: options.model,
          costPerInputToken: options.costPerInputToken,
          costPerOutputToken: options.costPerOutputToken,
        },
        functionKwargs: { args },
      },
      options.metrics,
    );
    return tracer.trace(async () => {
      const result = options.fn(...args);
      const finalResult = result instanceof Promise ? await result : result;
      return finalResult;
    });
  };
}

function ObserveRetriever<Args extends any[], T>(
  options: RetrieverOptions<Args, T>,
): (...args: Args) => Promise<T> {
  return async (...args: Args): Promise<T> => {
    const defaultName = options.name || options.fn.name || "Unknown";
    const tracer = new Tracer(
      SpanType.RETRIEVER,
      defaultName,
      options.metricCollection,
      {
        observeKwargs: {
          name: options.name,
          embedder: options.embedder,
        },
        functionKwargs: { args },
      },
      options.metrics,
    );
    return tracer.trace(async () => {
      const result = options.fn(...args);
      const finalResult = result instanceof Promise ? await result : result;
      return finalResult;
    });
  };
}

function ObserveTool<Args extends any[], T>(
  options: ToolOptions<Args, T>,
): (...args: Args) => Promise<T> {
  return async (...args: Args): Promise<T> => {
    const defaultName = options.name || options.fn.name || "Unknown";
    const tracer = new Tracer(
      SpanType.TOOL,
      defaultName,
      options.metricCollection,
      {
        observeKwargs: {
          name: defaultName,
          description: options.description,
        },
        functionKwargs: { args },
      },
      options.metrics,
    );
    return tracer.trace(async () => {
      const result = options.fn(...args);
      const finalResult = result instanceof Promise ? await result : result;
      return finalResult;
    });
  };
}

function ObserveCustom<Args extends any[], T>(
  options: CustomOptions<Args, T>,
): (...args: Args) => Promise<T> {
  return async (...args: Args): Promise<T> => {
    const defaultName = options.name || options.fn.name || "Unknown";
    const tracer = new Tracer(
      options.type,
      defaultName,
      options.metricCollection,
      {
        observeKwargs: {
          name: options.name,
        },
        functionKwargs: { args },
      },
      options.metrics,
    );
    return tracer.trace(async () => {
      const result = options.fn(...args);
      const finalResult = result instanceof Promise ? await result : result;
      return finalResult;
    });
  };
}

export function observe<Args extends any[], T>(options: {
  type?: SpanType | string;
  name?: string;
  metricCollection?: string;
  metrics?: BaseMetric[];
  model?: string;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  embedder?: string;
  description?: string;
  availableTools?: string[];
  agentHandoffs?: string[];
  /** Only observe when there is an active parent span. */
  _dropIfRoot?: boolean;
  /** Only observe when `CONFIDENT_TRACE_INTERNAL` is on. */
  _internal?: boolean;
  fn: (...args: Args) => T | Promise<T>;
}): (...args: Args) => Promise<T> {
  const { type, fn, _dropIfRoot, _internal, ...rest } = options;

  const observed = routeObserve<Args, T>(type, fn, rest);
  if (!_dropIfRoot && !_internal) return observed;

  return async (...args: Args): Promise<T> => {
    const skip =
      (_internal && !isTraceInternalEnabled()) ||
      (_dropIfRoot && getCurrentSpan() === undefined);
    return skip ? await fn(...args) : await observed(...args);
  };
}

function routeObserve<Args extends any[], T>(
  type: SpanType | string | undefined,
  fn: (...args: Args) => T | Promise<T>,
  rest: {
    name?: string;
    metricCollection?: string;
    metrics?: BaseMetric[];
    model?: string;
    costPerInputToken?: number;
    costPerOutputToken?: number;
    embedder?: string;
    description?: string;
    availableTools?: string[];
    agentHandoffs?: string[];
  },
): (...args: Args) => Promise<T> {
  // Route to the appropriate specialized observe function based on type
  if (type === SpanType.AGENT) {
    return ObserveAgent({
      fn,
      name: rest.name,
      metricCollection: rest.metricCollection,
      metrics: rest.metrics,
      availableTools: rest.availableTools,
      agentHandoffs: rest.agentHandoffs,
    });
  } else if (type === SpanType.LLM) {
    return ObserveLLM({
      fn,
      name: rest.name,
      metricCollection: rest.metricCollection,
      metrics: rest.metrics,
      model: rest.model,
      costPerInputToken: rest.costPerInputToken,
      costPerOutputToken: rest.costPerOutputToken,
    });
  } else if (type === SpanType.RETRIEVER) {
    return ObserveRetriever({
      fn,
      name: rest.name,
      metricCollection: rest.metricCollection,
      metrics: rest.metrics,
      embedder: rest.embedder,
    });
  } else if (type === SpanType.TOOL) {
    return ObserveTool({
      fn,
      name: rest.name,
      metricCollection: rest.metricCollection,
      metrics: rest.metrics,
      description: rest.description,
    });
  } else {
    // For custom types
    return ObserveCustom({
      fn,
      name: rest.name,
      metricCollection: rest.metricCollection,
      metrics: rest.metrics,
      type: "base",
    });
  }
}

////////////////////////////////////////////////////////
/// Update Current Span and Trace ///////////////////////
////////////////////////////////////////////////////////

export interface UpdateCurrentSpanParams {
  name?: string;
  input?: any;
  output?: any;
  testCase?: LLMTestCase;
  metadata?: Record<string, any>;
  retrievalContext?: string[];
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  expectedOutput?: string;
  context?: string[];
  metricCollection?: string;
  metrics?: BaseMetric[];
}

export const updateCurrentSpan = ({
  name,
  input,
  output,
  testCase,
  metadata,
  retrievalContext,
  toolsCalled,
  expectedTools,
  expectedOutput,
  context,
  metricCollection,
  metrics,
}: UpdateCurrentSpanParams) => {
  // Confident AI gets the update too when confident-trace exports the span.
  otelHooks.updateExported?.("span", {
    name,
    input,
    output,
    testCase,
    metadata,
    retrievalContext,
    toolsCalled,
    expectedTools,
    expectedOutput,
    context,
    metricCollection,
  });
  const currentSpan = activeSpan();
  if (!currentSpan) return;

  if (testCase) {
    currentSpan.input = testCase.input;
    currentSpan.output = testCase.actualOutput;
    currentSpan.retrievalContext = resolveRetrievalContext(
      testCase.retrievalContext,
    );
    currentSpan.toolsCalled = testCase.toolsCalled;
    currentSpan.expectedTools = testCase.expectedTools;
    currentSpan.context = testCase.context;
    currentSpan.expectedOutput = testCase.expectedOutput;
  }
  if (metadata) {
    currentSpan.metadata = metadata;
  }
  if (name) {
    currentSpan.name = name;
  }
  if (input) {
    currentSpan.input = input;
  }
  if (output) {
    currentSpan.output = output;
  }
  if (retrievalContext) {
    currentSpan.retrievalContext = retrievalContext;
  }
  if (toolsCalled) {
    currentSpan.toolsCalled = toolsCalled;
  }
  if (expectedTools) {
    currentSpan.expectedTools = expectedTools;
  }
  if (expectedOutput) {
    currentSpan.expectedOutput = expectedOutput;
  }
  if (context) {
    currentSpan.context = context;
  }
  if (metricCollection) {
    currentSpan.metricCollection = metricCollection;
  }
  if (metrics) {
    currentSpan.metrics = metrics;
  }
};

export interface UpdateCurrentTraceParams {
  tags?: string[];
  metadata?: Record<string, any>;
  threadId?: string;
  userId?: string;
  testCaseId?: string;
  testRunId?: string;
  turnId?: string;
  input?: any;
  output?: any;
  name?: string;
  testCase?: LLMTestCase;
  retrievalContext?: string[];
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  expectedOutput?: string;
  context?: string[];
  metricCollection?: string;
  metrics?: BaseMetric[];
  confidentApiKey?: string;
  drop?: boolean;
}

export const updateCurrentTrace = ({
  tags,
  metadata,
  threadId,
  userId,
  testCaseId,
  testRunId,
  turnId,
  input,
  output,
  name,
  testCase,
  retrievalContext,
  toolsCalled,
  expectedTools,
  expectedOutput,
  context,
  metricCollection,
  metrics,
  confidentApiKey,
  drop,
}: UpdateCurrentTraceParams) => {
  // Confident AI gets the update too when confident-trace exports the trace.
  otelHooks.updateExported?.("trace", {
    tags,
    metadata,
    threadId,
    userId,
    testCaseId,
    testRunId,
    turnId,
    input,
    output,
    name,
    testCase,
    retrievalContext,
    toolsCalled,
    expectedTools,
    expectedOutput,
    context,
    metricCollection,
    confidentApiKey,
  });
  if (drop) otelHooks.drop?.(true);
  const currentTrace = activeTrace();
  if (!currentTrace) return;

  if (testCase !== undefined) {
    currentTrace.input = testCase.input;
    currentTrace.output = testCase.actualOutput;
    currentTrace.retrievalContext = resolveRetrievalContext(
      testCase.retrievalContext,
    );
    currentTrace.toolsCalled = testCase.toolsCalled;
    currentTrace.expectedTools = testCase.expectedTools;
    currentTrace.context = testCase.context;
    currentTrace.expectedOutput = testCase.expectedOutput;
  }
  if (tags !== undefined) {
    currentTrace.tags = tags;
  }
  if (metadata !== undefined) {
    currentTrace.metadata = metadata;
  }
  if (threadId !== undefined) {
    currentTrace.threadId = threadId;
  }
  if (userId !== undefined) {
    currentTrace.userId = userId;
  }
  if (testCaseId !== undefined) {
    currentTrace.testCaseId = testCaseId;
  }
  if (testRunId !== undefined) {
    currentTrace.testRunId = testRunId;
  }
  if (turnId !== undefined) {
    currentTrace.turnId = turnId;
  }
  if (input !== undefined) {
    currentTrace.input = input;
  }
  if (output !== undefined) {
    currentTrace.output = output;
  }
  if (name !== undefined) {
    currentTrace.name = name;
  }
  if (retrievalContext !== undefined) {
    currentTrace.retrievalContext = retrievalContext;
  }
  if (toolsCalled !== undefined) {
    currentTrace.toolsCalled = toolsCalled;
  }
  if (expectedTools !== undefined) {
    currentTrace.expectedTools = expectedTools;
  }
  if (expectedOutput !== undefined) {
    currentTrace.expectedOutput = expectedOutput;
  }
  if (context !== undefined) {
    currentTrace.context = context;
  }
  if (metricCollection !== undefined) {
    currentTrace.metricCollection = metricCollection;
  }
  if (metrics !== undefined) {
    currentTrace.metrics = metrics;
  }
  if (confidentApiKey !== undefined) {
    currentTrace.confidentApiKey = confidentApiKey;
  }
  if (drop !== undefined) {
    currentTrace.drop = drop;
  }
};

interface UpdateLLMSpanParams {
  model?: string;
  costPerInputToken?: number;
  costPerOutputToken?: number;
  inputTokenCount?: number;
  outputTokenCount?: number;
  tokenIntervals?: number;
  prompt?: Prompt;
}

export const updateLlmSpan = ({
  model,
  costPerInputToken,
  costPerOutputToken,
  inputTokenCount,
  outputTokenCount,
  tokenIntervals,
  prompt,
}: UpdateLLMSpanParams) => {
  otelHooks.updateExported?.("span", {
    model,
    costPerInputToken,
    costPerOutputToken,
    inputTokenCount,
    outputTokenCount,
    prompt,
  });
  const currentSpan = activeSpan();
  if (!currentSpan) {
    return;
  }

  // TODO(tanay): check for instanceof LlmSpan and exit if not
  const llmSpan = currentSpan as LlmSpan;

  if (model !== undefined) {
    llmSpan.model = model;
  }
  if (costPerInputToken !== undefined) {
    llmSpan.costPerInputToken = costPerInputToken;
  }
  if (costPerOutputToken !== undefined) {
    llmSpan.costPerOutputToken = costPerOutputToken;
  }
  if (inputTokenCount !== undefined) {
    llmSpan.inputTokenCount = inputTokenCount;
  }
  if (outputTokenCount !== undefined) {
    llmSpan.outputTokenCount = outputTokenCount;
  }
  if (tokenIntervals !== undefined) {
    llmSpan.tokenInterval = tokenIntervals;
  }
  if (prompt !== undefined) {
    llmSpan.prompt = prompt;
    llmSpan.promptAlias = prompt._alias!;
    llmSpan.promptCommitHash = prompt.hash!;
    llmSpan.promptVersion = prompt.version!;
    llmSpan.promptLabel = prompt.label!;
  }
};

interface UpdateRetrieverSpanParams {
  embedder?: string;
  topK?: number;
  chunkSize?: number;
}

export const updateRetrieverSpan = ({
  embedder,
  topK,
  chunkSize,
}: UpdateRetrieverSpanParams) => {
  otelHooks.updateExported?.("span", { embedder, topK, chunkSize });
  const currentSpan = activeSpan();

  if (!currentSpan) {
    return;
  }

  // TODO(tanay): check for instanceof RetrieverSpan and exit if not
  const retrieverSpan = currentSpan as RetrieverSpan;

  if (embedder !== undefined) {
    retrieverSpan.embedder = embedder;
  }
  if (topK !== undefined) {
    retrieverSpan.topK = topK;
  }
  if (chunkSize !== undefined) {
    retrieverSpan.chunkSize = chunkSize;
  }
};
