/**
 * Local component-level evals for the OpenInference integration.
 *
 * Drives `OpenInferenceSpanProcessor` with synthetic OTel spans — no model call,
 * no network, no exporter.
 */
import { OpenInferenceSpanProcessor } from "../../../src/integrations/openinference/processor";
import { OpenInferenceFilterProcessor } from "../../../src/integrations/openinference";
import { BaseMetric } from "../../../src/metrics/base-metrics";
import {
  SpanType,
  getCurrentSpan,
  setCurrentSpan,
  setCurrentTrace,
  traceManager,
} from "../../../src/tracing/tracing";
import {
  isTraceOtelImplicit,
  ROUTE_TO_REST_ATTRIBUTE,
} from "../../../src/tracing/otel-routing";
import {
  nextLlmSpan,
  nextToolSpan,
  updateCurrentSpan,
} from "../../../src/tracing";

class StubMetric extends BaseMetric {
  constructor(private label: string) {
    super(0.5, { showIndicator: false });
    this.requiredParams = [];
  }
  get name(): string {
    return this.label;
  }
  async measure(): Promise<number> {
    this.score = 1;
    this.success = true;
    return 1;
  }
  isSuccessful(): boolean {
    return true;
  }
}

/** A minimal stand-in for an OTel span carrying OpenInference attributes. */
function oiSpan(spanId: string, kind: string, parentSpanId?: string): any {
  const attributes: Record<string, any> = {
    "openinference.span.kind": kind,
  };
  return {
    name: `oi.${kind.toLowerCase()}`,
    parentSpanId,
    attributes,
    spanContext: () => ({ spanId, traceId: "otel-trace-1" }),
    setAttribute(key: string, value: any) {
      attributes[key] = value;
      return this;
    },
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
  };
}

describe("OpenInference local component evals", () => {
  let processor: OpenInferenceSpanProcessor;
  let endEvaluation: (() => void) | undefined;

  beforeEach(() => {
    traceManager.clearTraces();
    traceManager.configure({ tracingEnabled: true });
    setCurrentSpan(null);
    setCurrentTrace(null);
    processor = new OpenInferenceSpanProcessor({});
  });

  afterEach(() => {
    endEvaluation?.();
    endEvaluation = undefined;
    setCurrentSpan(null);
    setCurrentTrace(null);
  });

  it("builds no local span for a bare caller (routes to OTLP)", () => {
    const span = oiSpan("s1", "LLM");
    processor.onStart(span, {} as any);

    expect(span.attributes[ROUTE_TO_REST_ATTRIBUTE]).toBeUndefined();
    expect(traceManager.getSpanByUuid("s1")).toBeUndefined();
  });

  it("builds a local span while a local eval pipeline is active", () => {
    endEvaluation = traceManager.beginEvaluation();

    const span = oiSpan("s2", "LLM");
    processor.onStart(span, {} as any);

    expect(span.attributes[ROUTE_TO_REST_ATTRIBUTE]).toBe(true);
    expect(traceManager.getSpanByUuid("s2")?.type).toBe(SpanType.LLM);
  });

  it("opens an implicit trace for the root when the caller has none", () => {
    endEvaluation = traceManager.beginEvaluation();

    processor.onStart(oiSpan("s3", "AGENT"), {} as any);

    const trace = traceManager.getAllTraces().at(-1)!;
    expect(isTraceOtelImplicit(trace)).toBe(true);
    expect(trace.rootSpans.map((s) => s.uuid)).toContain("s3");
  });

  it("applies metrics staged with nextLlmSpan", async () => {
    endEvaluation = traceManager.beginEvaluation();
    const metric = new StubMetric("llm-metric");

    await nextLlmSpan({ metrics: [metric] }, async () => {
      processor.onStart(oiSpan("s4", "LLM"), {} as any);
    });

    expect(traceManager.getSpanByUuid("s4")?.metrics).toEqual([metric]);
  });

  it("applies metrics staged with nextToolSpan to a tool span", async () => {
    endEvaluation = traceManager.beginEvaluation();
    const metric = new StubMetric("tool-metric");

    await nextToolSpan({ metrics: [metric] }, async () => {
      processor.onStart(oiSpan("s5", "TOOL"), {} as any);
    });

    const span = traceManager.getSpanByUuid("s5");
    expect(span?.type).toBe(SpanType.TOOL);
    expect(span?.metrics).toEqual([metric]);
  });

  it("reaches the span from a tool body via updateCurrentSpan", () => {
    endEvaluation = traceManager.beginEvaluation();
    const metric = new StubMetric("from-tool-body");

    processor.onStart(oiSpan("s6", "TOOL"), {} as any);
    expect(getCurrentSpan()?.uuid).toBe("s6");

    updateCurrentSpan({ metrics: [metric], expectedOutput: "Sunny" });

    expect(traceManager.getSpanByUuid("s6")?.metrics).toEqual([metric]);
    expect(traceManager.getSpanByUuid("s6")?.expectedOutput).toBe("Sunny");
  });

  it("restores the previous current span when a span ends", () => {
    endEvaluation = traceManager.beginEvaluation();

    processor.onStart(oiSpan("s7", "AGENT"), {} as any);
    const inner = oiSpan("s8", "TOOL", "s7");
    processor.onStart(inner, {} as any);
    expect(getCurrentSpan()?.uuid).toBe("s8");

    processor.onEnd(inner as any);
    expect(getCurrentSpan()?.uuid).toBe("s7");
  });
});

describe("OpenInference export routing", () => {
  it("skips OTLP export for spans routed to REST", () => {
    const exported: string[] = [];
    const underlying = {
      onStart: () => {},
      onEnd: (s: any) => exported.push(s.name),
      forceFlush: async () => {},
      shutdown: async () => {},
    };
    const filter = new OpenInferenceFilterProcessor(underlying as any);

    const restSpan = oiSpan("s9", "LLM");
    restSpan.attributes[ROUTE_TO_REST_ATTRIBUTE] = true;
    filter.onEnd(restSpan as any);
    expect(exported).toEqual([]);

    const otlpSpan = oiSpan("s10", "LLM");
    filter.onEnd(otlpSpan as any);
    expect(exported).toEqual(["oi.llm"]);
  });
});
