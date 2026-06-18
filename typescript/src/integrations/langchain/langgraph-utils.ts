import { traceManager } from "../../tracing";
import { setCurrentTrace } from "../../tracing/tracing";

export interface TraceInitFields {
  name?: string;
  tags?: string[];
  metadata?: Record<string, any>;
  threadId?: string;
  userId?: string;
  testCaseId?: string;
  turnId?: string;
}

/**
 * Tracks the run_id/parent_run_id hierarchy for a single CallbackHandler instance
 * and resolves each run's owning trace + parent span without depending on
 * AsyncLocalStorage. Also owns lazy, per-run trace creation: a handler baked into
 * a long-lived graph (e.g. exported via langgraph.json) is reused across many
 * server requests, so the trace must be created per run rather than once in the
 * handler constructor.
 */
export class RunHierarchyTracker {
  private runIdToSpanUuid = new Map<string, string>();
  private runIdToParentRunId = new Map<string, string | undefined>();
  private runIdToTraceUuid = new Map<string, string>();
  private threadIdToTraceUuid = new Map<string, string>();

  constructor(private initFields: TraceInitFields) {}

  /** Record linkage for a run (call for every run, span-bearing or not). */
  recordRun(runId: string, parentRunId: string | undefined, traceUuid: string) {
    this.runIdToParentRunId.set(runId, parentRunId);
    this.runIdToTraceUuid.set(runId, traceUuid);
  }

  /** Mark that this run owns a span (span uuid == run id). */
  recordSpan(runId: string) {
    this.runIdToSpanUuid.set(runId, runId);
  }

  /** Drop a run's bookkeeping once it has ended (keeps maps bounded). */
  cleanupRun(runId: string) {
    this.runIdToSpanUuid.delete(runId);
    this.runIdToParentRunId.delete(runId);
    this.runIdToTraceUuid.delete(runId);
  }

  getTraceUuid(runId: string): string | undefined {
    return this.runIdToTraceUuid.get(runId);
  }

  /** Resolve the trace + parent span for a run from run_id / parent_run_id. */
  resolveContext(
    runId: string,
    parentRunId?: string,
  ): { traceUuid: string; parentUuid?: string } {
    if (parentRunId) {
      const parentSpanUuid = this.resolveParentSpanUuid(parentRunId);
      if (parentSpanUuid) {
        const parentSpan = traceManager.getSpanByUuid(parentSpanUuid);
        if (parentSpan) {
          return {
            traceUuid: parentSpan.traceUuid,
            parentUuid: parentSpan.uuid,
          };
        }
      }
      const ancestorTrace = this.resolveTraceUuidFromAncestors(parentRunId);
      if (ancestorTrace) {
        return { traceUuid: ancestorTrace, parentUuid: undefined };
      }
    }
    const trace = this.ensureTrace();
    return { traceUuid: trace.uuid, parentUuid: undefined };
  }

  // Walk parent_run_id up the chain to the nearest run that owns a live span.
  private resolveParentSpanUuid(parentRunId?: string): string | undefined {
    let pr = parentRunId;
    const seen = new Set<string>();
    while (pr && !seen.has(pr)) {
      seen.add(pr);
      const spanUuid = this.runIdToSpanUuid.get(pr);
      if (spanUuid && traceManager.getSpanByUuid(spanUuid)) {
        return spanUuid;
      }
      pr = this.runIdToParentRunId.get(pr);
    }
    return undefined;
  }

  // Walk parent_run_id up the chain to the nearest run with a still-active trace.
  private resolveTraceUuidFromAncestors(
    parentRunId?: string,
  ): string | undefined {
    let pr = parentRunId;
    const seen = new Set<string>();
    while (pr && !seen.has(pr)) {
      seen.add(pr);
      const traceUuid = this.runIdToTraceUuid.get(pr);
      if (traceUuid && traceManager.getTraceByUuid(traceUuid)) {
        return traceUuid;
      }
      pr = this.runIdToParentRunId.get(pr);
    }
    return undefined;
  }

  private ensureTrace() {
    const threadId = this.initFields.threadId;
    if (threadId) {
      const existing = this.threadIdToTraceUuid.get(threadId);
      if (existing) {
        const active = traceManager.getTraceByUuid(existing);
        if (active) return active;
      }
    }
    const trace = traceManager.startNewTrace();
    this.applyInitFields(trace);
    if (threadId) this.threadIdToTraceUuid.set(threadId, trace.uuid);
    setCurrentTrace(trace);
    return trace;
  }

  private applyInitFields(trace: any) {
    const f = this.initFields;
    if (f.name !== undefined) trace.name = f.name;
    if (f.tags !== undefined) trace.tags = f.tags;
    if (f.metadata !== undefined) trace.metadata = f.metadata;
    if (f.threadId !== undefined) trace.threadId = f.threadId;
    if (f.userId !== undefined) trace.userId = f.userId;
    if (f.testCaseId !== undefined) trace.testCaseId = f.testCaseId;
    if (f.turnId !== undefined) trace.turnId = f.turnId;
  }
}
