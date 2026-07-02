import { AsyncLocalStorage } from "async_hooks";
import console from "console";

import { getSettings } from "../config/settings";
import { isConfident, wait } from "../utils";
import {
  tracingEnabled,
  validateEnvironment,
  validateSamplingRate,
  toZodCompatibleISO,
  Environment,
} from "./utils";

import { Api, Endpoints, HttpMethods } from "../confident/api";
import { LLMTestCase, ToolCall, resolveRetrievalContext } from "../test-case";
import type { BaseMetric } from "../metrics/base-metrics";
import type { MetricData } from "../evaluate/types";
import { Prompt } from "../prompt";
import { SpanApiType, BaseApiSpan, TraceApi, TraceSpanApiStatus } from "./api";
import { TraceWorkerStatus, printTraceStatus } from "./logging";

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
  maxTraces?: number;
}

export class TraceManager {
  private traces: Trace[] = [];
  private activeTraces: Map<string, Trace> = new Map(); // Map of traceUuid to Trace
  private activeSpans: Map<string, BaseSpan> = new Map(); // Map of spanUuid to BaseSpan
  private traceSpanCounters: Map<string, number> = new Map(); // Map of traceUuid to active span count

  // Queue for trace posting
  private traceQueue: Trace[] = [];
  private isProcessing: boolean = false;
  private minInterval: number = 200; // Minimum time between API calls (milliseconds)
  private lastPostTime: number = 0;
  private inFlightRequests: Set<Promise<any>> = new Set();
  // private flushEnabled: boolean;

  // Configuration
  private environment: Environment;
  private samplingRate: number;
  private customMaskFn: MaskFunction | null = null;
  private evaluating: boolean = false;
  /** When set, completed traces are captured here (for local eval) instead of posted. */
  private traceCaptureSink?: (trace: Trace) => void;
  private confidentApiKey: string = "";
  private tracingEnabled: boolean = true;
  private maxTraces: number = 0; // 0 = unlimited

  /** Register/clear a sink that receives each completed trace (used by `evalsIterator`). */
  public setTraceCaptureSink(sink?: (trace: Trace) => void): void {
    this.traceCaptureSink = sink;
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
    const add = (k: string, v: unknown) => {
      if (v != null) dict[k] = v;
    };
    add("name", span.name);
    add("type", span.type);
    add("input", span.input);
    add("output", span.output);
    add("error", span.error);
    add("expectedOutput", span.expectedOutput);
    add("context", span.context);
    add("retrievalContext", span.retrievalContext);
    add("toolsCalled", span.toolsCalled);
    add("expectedTools", span.expectedTools);
    add("model", (span as { model?: unknown }).model);
    add(
      "availableTools",
      (span as { availableTools?: unknown }).availableTools,
    );
    dict.children = span.children.map((child) =>
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
    if (config.maxTraces !== undefined) {
      this.maxTraces = config.maxTraces;
    }
  }

  public startNewTrace(): Trace {
    const traceUuid = crypto.randomUUID();
    const newTrace: Trace = {
      uuid: traceUuid,
      rootSpans: [],
      status: TraceSpanStatus.IN_PROGRESS,
      startTime: new Date(),
      endTime: undefined,
    };
    this.activeTraces.set(traceUuid, newTrace);
    this.traces.push(newTrace);
    if (this.maxTraces > 0 && this.traces.length > this.maxTraces) {
      const evicted = this.traces.splice(
        0,
        this.traces.length - this.maxTraces,
      );
      for (const t of evicted) {
        this.activeTraces.delete(t.uuid);
      }
    }
    return newTrace;
  }

  public endTrace(traceUuid: string): void {
    const trace = this.activeTraces.get(traceUuid);
    if (trace) {
      trace.endTime = new Date();
      if (trace.status === TraceSpanStatus.IN_PROGRESS) {
        trace.status = TraceSpanStatus.SUCCESS;
      }
      if (this.traceCaptureSink) {
        this.traceCaptureSink(trace);
      } else if (!this.evaluating) {
        this.postTrace(trace);
      } else {
        trace.rootSpans = [trace.rootSpans[0].children[0]];
        for (const rootSpan of trace.rootSpans) {
          rootSpan.parentUuid = undefined;
        }
      }
      this.activeTraces.delete(traceUuid);
    }
  }

  public setTraceStatus(traceUuid: string, status: TraceSpanStatus): void {
    const trace = this.activeTraces.get(traceUuid);
    if (trace) {
      trace.status = status;
    }
  }

  public addSpan(span: BaseSpan): void {
    this.activeSpans.set(span.uuid, span);
    this.traceSpanCounters.set(
      span.traceUuid,
      (this.traceSpanCounters.get(span.traceUuid) ?? 0) + 1,
    );
  }

  public removeSpan(spanUuid: string): void {
    const span = this.activeSpans.get(spanUuid);
    if (span) {
      this.activeSpans.delete(spanUuid);
      const count = this.traceSpanCounters.get(span.traceUuid);
      if (count !== undefined) {
        if (count <= 1) {
          this.traceSpanCounters.delete(span.traceUuid);
        } else {
          this.traceSpanCounters.set(span.traceUuid, count - 1);
        }
      }
    }
  }

  public getActiveSpanCount(traceUuid: string): number {
    return this.traceSpanCounters.get(traceUuid) ?? 0;
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
  public updateSpanInTrace(span: BaseSpan): void {
    const traceUuid = span.traceUuid;
    const trace = this.activeTraces.get(traceUuid);
    if (!trace) {
      throw new Error(
        `Trace with UUID ${traceUuid} does not exist. A span must have a valid trace.`,
      );
    }
    const existing = this.getSpanByUuid(span.uuid);
    if (!existing) {
      throw new Error(
        `Span with UUID ${span.uuid} not found in trace ${traceUuid}.`,
      );
    }

    Object.assign(existing, {
      name: span.name,
      startTime: span.startTime,
      endTime: span.endTime,
      status: span.status,
      error: span.error,
      input: span.input,
      output: span.output,
      metadata: span.metadata,
      metricCollection: span.metricCollection,
      expectedOutput: span.expectedOutput,
      retrievalContext: span.retrievalContext,
      context: span.context,
      toolsCalled: span.toolsCalled,
      expectedTools: span.expectedTools,
    });

    if (span.type === SpanType.LLM) {
      const llmSpan = span as LlmSpan;
      const existingLlmSpan = existing as LlmSpan;
      existingLlmSpan.model = llmSpan.model;
      existingLlmSpan.costPerInputToken = llmSpan.costPerInputToken;
      existingLlmSpan.costPerOutputToken = llmSpan.costPerOutputToken;
      existingLlmSpan.inputTokenCount = llmSpan.inputTokenCount;
      existingLlmSpan.promptAlias = llmSpan.promptAlias;
      existingLlmSpan.promptCommitHash = llmSpan.promptCommitHash;
      existingLlmSpan.promptVersion = llmSpan.promptVersion;
      existingLlmSpan.promptLabel = llmSpan.promptLabel;
      existingLlmSpan.outputTokenCount = llmSpan.outputTokenCount;
    } else if (span.type === SpanType.RETRIEVER) {
      const retrieverSpan = span as RetrieverSpan;
      const existingRetrieverSpan = existing as RetrieverSpan;
      existingRetrieverSpan.embedder = retrieverSpan.embedder;
      existingRetrieverSpan.topK = retrieverSpan.topK;
      existingRetrieverSpan.chunkSize = retrieverSpan.chunkSize;
    }
  }

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
    this.traceSpanCounters.clear();
    // Reset async-local trace/span pointers so callers (notably test
    // `beforeEach`) get a fresh context. Without this, AsyncLocalStorage
    // leaks a stale trace UUID across tests and span attachment throws.
    setCurrentTrace(null);
    setCurrentSpan(null);
  }

  private shouldSampleTrace(): boolean {
    const randomNumber = Math.random();
    if (randomNumber > this.samplingRate) {
      const rateStr = this.samplingRate.toFixed(2);
      printTraceStatus(
        TraceWorkerStatus.SUCCESS,
        `Skipped posting trace due to sampling rate (${rateStr})`,
      );
      return false;
    }
    return true;
  }

  public postTrace(trace: Trace): string | undefined {
    if (!tracingEnabled() || !this.tracingEnabled || trace.drop) {
      return;
    }

    const hasApiKey =
      trace.confidentApiKey || this.confidentApiKey || isConfident();
    if (!hasApiKey) {
      printTraceStatus(
        TraceWorkerStatus.FAILURE,
        "No Confident API key found. Skipping trace posting.",
      );
      return;
    }

    if (!this.shouldSampleTrace()) {
      return;
    }

    this.traceQueue.push(trace);
    if (!this.isProcessing) {
      this.processTraceQueue();
    }
    return "ok";
  }

  /** Wait for all queued + in-flight trace posts to finish (for scripts/eval loops). */
  public async flush(): Promise<void> {
    while (this.traceQueue.length > 0 || this.isProcessing) {
      await wait(20);
    }
    await Promise.allSettled([...this.inFlightRequests]);
  }

  private async processTraceQueue(): Promise<void> {
    this.isProcessing = true;
    try {
      while (this.traceQueue.length > 0) {
        this.cleanFinishedRequests();
        const trace = this.traceQueue.shift();
        if (!trace) {
          continue;
        }

        const now = Date.now();
        const timeSinceLast = now - this.lastPostTime;
        if (timeSinceLast < this.minInterval) {
          await wait(this.minInterval - timeSinceLast);
        }

        this.lastPostTime = Date.now();
        const sendTracePromise = this.sendTrace(trace).catch((_error) => {
          printTraceStatus(
            TraceWorkerStatus.WARNING,
            `Processing continues despite error with trace ${trace.uuid}`,
          );
        });
        this.inFlightRequests.add(sendTracePromise);
      }
    } catch (error) {
      printTraceStatus(
        TraceWorkerStatus.FAILURE,
        "Error in trace queue processing",
        error instanceof Error ? error.message : String(error),
      );
    } finally {
      this.isProcessing = false;
      if (this.traceQueue.length > 0) {
        this.processTraceQueue();
      }
    }
  }

  // Clean up finished requests
  private cleanFinishedRequests(): void {
    for (const request of this.inFlightRequests) {
      if (
        (request as any).status === "fulfilled" ||
        (request as any).status === "rejected"
      ) {
        this.inFlightRequests.delete(request);
      }
    }
  }

  // Send a trace to the Confident API
  private async sendTrace(trace: Trace): Promise<void> {
    try {
      const traceApi = this.createTraceApi(trace);
      const apiKey = traceApi.confidentApiKey || this.confidentApiKey;
      const api = new Api(apiKey);
      const { confidentApiKey: _, ...traceApiBody } = traceApi;

      const response = await api.sendRequest(
        HttpMethods.POST,
        Endpoints.TRACES_ENDPOINT,
        traceApiBody,
      );
      const queueSize = this.traceQueue.length;
      const inFlightCount = this.inFlightRequests.size;
      const status = `(${queueSize} trace(s) remaining in queue, ${inFlightCount} in flight)`;
      // TODO(tanay): if env is testing, wait for log OR make test async
      if (this.environment !== Environment.TESTING) {
        printTraceStatus(
          TraceWorkerStatus.SUCCESS,
          `Successfully posted trace ${status}`,
          response?.link,
          this.environment,
        );
      }
    } catch (error) {
      const queueSize = this.traceQueue.length;
      const inFlightCount = this.inFlightRequests.size;
      const status = `(${queueSize} trace(s) remaining in queue, ${inFlightCount} in flight)`;
      printTraceStatus(
        TraceWorkerStatus.FAILURE,
        `Error posting trace ${status}`,
        error instanceof Error ? error.message : String(error),
      );

      throw error;
    }
  }

  public createTraceApi(trace: Trace): TraceApi {
    const baseSpans: BaseApiSpan[] = [];
    const agentSpans: BaseApiSpan[] = [];
    const llmSpans: BaseApiSpan[] = [];
    const retrieverSpans: BaseApiSpan[] = [];
    const toolSpans: BaseApiSpan[] = [];
    const stack = [...trace.rootSpans];
    while (stack.length > 0) {
      const span = stack.pop()!;
      const apiSpan = this.convertSpanToApiSpan(span);
      if (apiSpan.type === SpanType.RETRIEVER && !apiSpan.embedder) {
        apiSpan.embedder = "default-embedder";
      }
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
      if (span.children.length > 0) {
        for (let i = span.children.length - 1; i >= 0; i--) {
          stack.push(span.children[i]);
        }
      }
    }
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
      metricsData: trace.metricsData,
      confidentApiKey: trace.confidentApiKey,
      status:
        trace.status === TraceSpanStatus.SUCCESS
          ? TraceSpanApiStatus.SUCCESS
          : TraceSpanApiStatus.ERROR,
    };
  }

  private convertSpanToApiSpan(span: BaseSpan): BaseApiSpan {
    const startTime = span.startTime
      ? toZodCompatibleISO(span.startTime)
      : toZodCompatibleISO(new Date());
    let endTime: string;
    if (!span.endTime || span.endTime < span.startTime) {
      endTime = startTime;
    } else {
      endTime = toZodCompatibleISO(span.endTime);
    }

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
      metadata: span.metadata,
    };

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

export class Tracer {
  private uuid: string;
  private traceUuid: string | null = null;
  private parentUuid: string | null = null;
  private startTime!: Date;
  private endTime!: Date;
  private status: TraceSpanStatus = TraceSpanStatus.IN_PROGRESS;
  private error: string | null = null;
  private spanType: SpanType | string;
  private name: string;
  private metricCollection?: string;
  private metrics?: BaseMetric[];
  private observeKwargs: Record<string, any>;
  private functionKwargs: Record<string, any>;
  private result: any = null;

  constructor(
    spanType: SpanType | string | null,
    funcName: string,
    metricCollection?: string,
    kwargs: Record<string, any> = {},
    metrics?: BaseMetric[],
  ) {
    this.uuid = crypto.randomUUID();
    this.observeKwargs = kwargs.observeKwargs || {};
    this.functionKwargs = kwargs.functionKwargs || {};

    // Initialize name from options or function name
    this.name = this.observeKwargs.name || funcName;
    this.metricCollection = metricCollection;
    this.metrics = metrics;

    // Initialize span type from parameter or name
    this.spanType = spanType === null ? this.name : spanType;
  }

  // Enter the tracer context, creating a new span and setting up parent-child relationships.
  public enter(): this {
    this.startTime = new Date();
    const currentSpan = getCurrentSpan();
    const currentTrace = getCurrentTrace();
    if (!currentSpan || !currentTrace) {
      console.warn("Tracer.enter() called outside of tracing context");
      return this;
    }
    this.parentUuid = currentSpan.parentUuid || null;
    this.traceUuid = currentTrace.uuid;
    const spanInstance = this.createSpanInstance();
    traceManager.addSpan(spanInstance);
    traceManager.addSpanToTrace(spanInstance);
    return this;
  }

  // Exit the tracer context, updating the span status and handling trace completion.
  public exit(excType?: any, excVal?: any, _excTb?: any): void {
    this.endTime = new Date();
    const span = getCurrentSpan();
    if (!span || span.uuid !== this.uuid) {
      console.error(
        `Mismatch exiting span. Expected: ${this.uuid}, got: ${
          span?.uuid || "None"
        }`,
      );
      return;
    }
    span.endTime = this.endTime;
    if (excType) {
      span.status = TraceSpanStatus.ERRORED;
      span.error = String(excVal);
    } else {
      span.status = TraceSpanStatus.SUCCESS;
    }

    if (span.input === undefined) {
      span.input = traceManager.mask(this.functionKwargs);
    } else {
      span.input = traceManager.mask(span.input);
    }

    if (span.output === undefined) {
      span.output = traceManager.mask(this.result);
    } else {
      span.output = traceManager.mask(span.output);
    }

    traceManager.updateSpanInTrace(span);
    traceManager.removeSpan(span.uuid);
    const trace = getCurrentTrace();
    const isRootSpan = span.parentUuid === undefined;
    if (trace && trace.uuid === span.traceUuid && isRootSpan) {
      if (span.input === undefined) {
        updateCurrentTrace({
          input: this.functionKwargs,
        });
      }
      if (span.output === undefined) {
        updateCurrentTrace({
          output: this.result,
        });
      }
      if (span.status === TraceSpanStatus.ERRORED) {
        trace.status = TraceSpanStatus.ERRORED;
      }
      const otherActiveSpans = traceManager.getActiveSpanCount(span.traceUuid);
      if (otherActiveSpans === 0) {
        traceManager.endTrace(span.traceUuid);
      }
    }
  }

  // Context manager for tracing spans - async version for TypeScript compatibility
  public async trace<T>(callback: () => T | Promise<T>): Promise<T> {
    const trace = getCurrentTrace() ?? traceManager.startNewTrace();
    const parentSpan = getCurrentSpan();

    this.startTime = new Date();
    this.parentUuid = parentSpan?.uuid ?? null;
    this.traceUuid = trace.uuid;

    const span = this.createSpanInstance();
    traceManager.addSpan(span);
    traceManager.addSpanToTrace(span);

    return await withTracingContext(span, trace, async () => {
      try {
        this.result = await callback();
        return this.result;
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.endTime = new Date();
        const currentSpan = getCurrentSpan();
        if (!currentSpan || currentSpan.uuid !== this.uuid) {
          console.error(
            `Mismatch exiting span. Expected: ${this.uuid}, got: ${
              currentSpan?.uuid || "None"
            }`,
          );
        } else {
          if (this.error) {
            currentSpan.status = TraceSpanStatus.ERRORED;
            currentSpan.error = this.error;
          } else {
            currentSpan.status = TraceSpanStatus.SUCCESS;
          }

          if (currentSpan.input === undefined) {
            currentSpan.input = traceManager.mask(this.functionKwargs);
          } else {
            currentSpan.input = traceManager.mask(currentSpan.input);
          }

          if (currentSpan.output === undefined) {
            currentSpan.output = traceManager.mask(this.result);
          } else {
            currentSpan.output = traceManager.mask(currentSpan.output);
          }

          traceManager.removeSpan(currentSpan.uuid);

          const currentTrace = getCurrentTrace();
          const isRootSpan = currentSpan.parentUuid === undefined;
          if (
            currentTrace &&
            currentTrace.uuid === currentSpan.traceUuid &&
            isRootSpan
          ) {
            if (currentSpan.input === undefined) {
              updateCurrentTrace({
                input: this.functionKwargs,
              });
            }
            if (currentSpan.output === undefined) {
              updateCurrentTrace({
                output: this.result,
              });
            }
            if (currentSpan.status === TraceSpanStatus.ERRORED) {
              currentTrace.status = TraceSpanStatus.ERRORED;
            }
            if (traceManager.getActiveSpanCount(currentSpan.traceUuid) === 0) {
              traceManager.endTrace(currentSpan.traceUuid);
            }
          }
        }
      }
    });
  }

  // Create a span instance based on the span type.
  private createSpanInstance(): BaseSpan {
    if (this.spanType === SpanType.AGENT) {
      return {
        uuid: this.uuid,
        traceUuid: this.traceUuid!,
        parentUuid: this.parentUuid || undefined,
        startTime: this.startTime,
        endTime: this.endTime,
        status: TraceSpanStatus.SUCCESS,
        children: [],
        name: this.name,
        input: undefined,
        output: undefined,
        metricCollection: this.metricCollection,
        metrics: this.metrics,
        type: this.spanType,
        metadata: undefined,
        expectedOutput: undefined,
        retrievalContext: undefined,
        context: undefined,
        toolsCalled: undefined,
        expectedTools: undefined,
        availableTools: this.observeKwargs.availableTools || [],
        agentHandoffs: this.observeKwargs.agentHandoffs || [],
      } as AgentSpan;
    }
    if (this.spanType === SpanType.LLM) {
      return {
        uuid: this.uuid,
        traceUuid: this.traceUuid!,
        parentUuid: this.parentUuid || undefined,
        startTime: this.startTime,
        endTime: this.endTime,
        status: TraceSpanStatus.SUCCESS,
        children: [],
        name: this.name,
        input: undefined,
        output: undefined,
        metricCollection: this.metricCollection,
        metrics: this.metrics,
        type: this.spanType,
        metadata: undefined,
        expectedOutput: undefined,
        retrievalContext: undefined,
        context: undefined,
        toolsCalled: undefined,
        expectedTools: undefined,
        model: this.observeKwargs.model,
      } as LlmSpan;
    }
    if (this.spanType === SpanType.RETRIEVER) {
      return {
        uuid: this.uuid,
        traceUuid: this.traceUuid!,
        parentUuid: this.parentUuid || undefined,
        startTime: this.startTime,
        endTime: this.endTime,
        status: TraceSpanStatus.SUCCESS,
        children: [],
        name: this.name,
        input: undefined,
        output: undefined,
        metricCollection: this.metricCollection,
        metrics: this.metrics,
        type: this.spanType,
        metadata: undefined,
        expectedOutput: undefined,
        retrievalContext: undefined,
        context: undefined,
        toolsCalled: undefined,
        expectedTools: undefined,
        embedder: this.observeKwargs.embedder,
      } as RetrieverSpan;
    }
    if (this.spanType === SpanType.TOOL) {
      return {
        uuid: this.uuid,
        traceUuid: this.traceUuid!,
        parentUuid: this.parentUuid || undefined,
        startTime: this.startTime,
        endTime: this.endTime,
        status: TraceSpanStatus.SUCCESS,
        children: [],
        name: this.name,
        input: undefined,
        output: undefined,
        metricCollection: this.metricCollection,
        metrics: this.metrics,
        type: this.spanType,
        metadata: undefined,
        expectedOutput: undefined,
        retrievalContext: undefined,
        context: undefined,
        toolsCalled: undefined,
        expectedTools: undefined,
        description: this.observeKwargs.description,
      } as ToolSpan;
    }
    return {
      uuid: this.uuid,
      traceUuid: this.traceUuid!,
      parentUuid: this.parentUuid || undefined,
      startTime: this.startTime,
      endTime: this.endTime,
      status: TraceSpanStatus.SUCCESS,
      children: [],
      name: this.name,
      input: undefined,
      output: undefined,
      metricCollection: this.metricCollection,
      metrics: this.metrics,
      type: this.spanType,
      metadata: undefined,
      expectedOutput: undefined,
      retrievalContext: undefined,
      context: undefined,
      toolsCalled: undefined,
      expectedTools: undefined,
    };
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
  fn: (...args: Args) => T | Promise<T>;
}): (...args: Args) => Promise<T> {
  const { type, fn, ...rest } = options;

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
  const currentSpan = getCurrentSpan();

  if (!currentSpan) {
    return;
  }

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
  const currentTrace = getCurrentTrace();

  if (!currentTrace) {
    return;
  }

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
  const currentSpan = getCurrentSpan();
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
  const currentSpan = getCurrentSpan();

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
