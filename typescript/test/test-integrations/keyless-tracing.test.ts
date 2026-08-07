/**
 * Every integration must be fully functional without a Confident AI API key.
 *
 * A key buys one thing: uploading the trace. Span construction, `next*Span`
 * staging, `updateCurrentSpan`, and local evals all read the in-process trace
 * and must not depend on being logged in. This mirrors the Python SDK, where
 * `DeepEvalInstrumentationSettings` documents the key as fully optional.
 *
 * The OTel-based integrations are the subtle ones: with no key there is no OTLP
 * exporter to send to, so their spans have to route in-process instead of being
 * handed to a transport that cannot exist.
 */
import { SpanType as MastraSpanType } from "@mastra/core/observability";
import { TracingEventType } from "@mastra/core/observability";
import type { AnyExportedSpan, TracingEvent } from "@mastra/core/observability";

import {
  createOpenInferenceProcessors,
  OpenInferenceSpanProcessor,
} from "../../src/integrations/openinference";
import { createDeepEvalProcessors } from "../../src/integrations/ai-sdk";
import { DeepEvalSpanProcessor } from "../../src/integrations/ai-sdk/processor";
import { DeepEvalExporter } from "../../src/integrations/mastra";
import { BaseMetric } from "../../src/metrics/base-metrics";
import { EvaluationDataset, Golden } from "../../src/dataset";
import { ROUTE_TO_REST_ATTRIBUTE } from "../../src/tracing/otel-routing";
import { nextLlmSpan } from "../../src/tracing";
import {
  setCurrentSpan,
  setCurrentTrace,
  SpanType,
  traceManager,
} from "../../src/tracing/tracing";

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
function oiSpan(spanId: string, kind: string): any {
  const attributes: Record<string, any> = { "openinference.span.kind": kind };
  return {
    name: `oi.${kind.toLowerCase()}`,
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

/** A minimal stand-in for an "ai.*" OTel span from the Vercel AI SDK. */
function aiSdkSpan(spanId: string, name: string): any {
  const attributes: Record<string, any> = {};
  return {
    name,
    attributes,
    spanContext: () => ({ spanId, traceId: "otel-trace-2" }),
    setAttribute(key: string, value: any) {
      attributes[key] = value;
      return this;
    },
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
  };
}

function mastraSpan(overrides: Partial<AnyExportedSpan>): AnyExportedSpan {
  return {
    id: "mastra-span-1",
    traceId: "mastra-trace-1",
    name: "llm",
    startTime: new Date(),
    attributes: {},
    isRootSpan: false,
    isEvent: false,
    type: MastraSpanType.MODEL_GENERATION,
    ...overrides,
  } as AnyExportedSpan;
}

describe("integrations work without CONFIDENT_API_KEY", () => {
  const savedKey = process.env.CONFIDENT_API_KEY;

  beforeEach(() => {
    delete process.env.CONFIDENT_API_KEY;
    traceManager.clearTraces();
    traceManager.configure({ tracingEnabled: true });
    setCurrentSpan(null);
    setCurrentTrace(null);
  });

  afterAll(() => {
    if (savedKey === undefined) delete process.env.CONFIDENT_API_KEY;
    else process.env.CONFIDENT_API_KEY = savedKey;
  });

  describe("OpenInference", () => {
    it("still installs the local span processor", () => {
      const processors = createOpenInferenceProcessors({});

      expect(processors).toHaveLength(1);
      expect(processors[0]).toBeInstanceOf(OpenInferenceSpanProcessor);
    });

    it("materialises spans in-process instead of routing to an absent OTLP exporter", () => {
      const processor = new OpenInferenceSpanProcessor(
        {},
        { otlpEnabled: false },
      );
      const span = oiSpan("oi-1", "LLM");

      processor.onStart(span, {} as any);

      expect(span.attributes[ROUTE_TO_REST_ATTRIBUTE]).toBe(true);
      expect(traceManager.getSpanByUuid("oi-1")?.type).toBe(SpanType.LLM);
    });

    it("drains next*Span staging onto the span", async () => {
      const processor = new OpenInferenceSpanProcessor(
        {},
        { otlpEnabled: false },
      );
      const metric = new StubMetric("llm-metric");

      await nextLlmSpan({ metrics: [metric] }, async () => {
        processor.onStart(oiSpan("oi-2", "LLM"), {} as any);
      });

      expect(traceManager.getSpanByUuid("oi-2")?.metrics).toEqual([metric]);
    });
  });

  describe("Vercel AI SDK", () => {
    it("still installs the local span processor", () => {
      const processors = createDeepEvalProcessors({});

      expect(processors).toHaveLength(1);
      expect(processors[0]).toBeInstanceOf(DeepEvalSpanProcessor);
    });

    it("materialises spans in-process instead of routing to an absent OTLP exporter", () => {
      const processor = new DeepEvalSpanProcessor({}, { otlpEnabled: false });
      const span = aiSdkSpan("ai-1", "ai.generateText");

      processor.onStart(span, {} as any);

      expect(span.attributes[ROUTE_TO_REST_ATTRIBUTE]).toBe(true);
      expect(traceManager.getSpanByUuid("ai-1")).toBeDefined();
    });
  });

  describe("Mastra", () => {
    it("does not disable itself", async () => {
      const exporter = new DeepEvalExporter({ name: "keyless" });

      await exporter.exportTracingEvent({
        type: TracingEventType.SPAN_STARTED,
        exportedSpan: mastraSpan({ id: "mastra-1" }),
      } as TracingEvent);

      expect(traceManager.getSpanByUuid("mastra-1")?.type).toBe(SpanType.LLM);
    });

    it("drains next*Span staging onto the span", async () => {
      const exporter = new DeepEvalExporter({ name: "keyless" });
      const metric = new StubMetric("llm-metric");

      await nextLlmSpan({ metrics: [metric] }, async () => {
        await exporter.exportTracingEvent({
          type: TracingEventType.SPAN_STARTED,
          exportedSpan: mastraSpan({ id: "mastra-2" }),
        } as TracingEvent);
      });

      expect(traceManager.getSpanByUuid("mastra-2")?.metrics).toEqual([metric]);
    });
  });

  it("scores an OTel integration's spans through evalsIterator", async () => {
    const processor = new OpenInferenceSpanProcessor({}, { otlpEnabled: false });
    const metric = new StubMetric("component-metric");
    const dataset = new EvaluationDataset({
      goldens: [new Golden({ input: "What's the weather in Tokyo?" })],
    });

    for await (const golden of dataset.evalsIterator({
      displayConfig: { showIndicator: false },
    })) {
      await nextLlmSpan(
        { metrics: [metric], input: (golden as Golden).input },
        async () => {
          const span = oiSpan("eval-1", "LLM");
          processor.onStart(span, {} as any);
          processor.onEnd(span);
        },
      );
    }

    const scored = dataset.evalResults.flatMap((r) => r.metricsData ?? []);
    expect(scored.map((m) => m.name)).toContain("component-metric");
    expect(scored.every((m) => m.success)).toBe(true);
  });
});
