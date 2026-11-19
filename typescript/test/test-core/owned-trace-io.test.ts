import { randomUUID } from "node:crypto";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";

import { DeepEvalExporter } from "@/integrations/mastra";
import {
  confidentTrace,
  ensureConfidentTrace,
  shutdownConfidentTrace,
} from "@/tracing/otel/setup";
import { type Trace, traceManager, updateCurrentTrace } from "@/tracing";

ensureConfidentTrace({ exporter: new InMemorySpanExporter() });
const { traceContext, withSpan } = confidentTrace();

const modelInput = "internal model input";
const modelOutput = "internal model output";
const requestInput = "request input";

async function captureTrace(run: () => unknown): Promise<Trace> {
  const traces: Trace[] = [];
  const unsubscribe = traceManager.addTraceCaptureSink((trace) =>
    traces.push(trace),
  );
  try {
    await run();
  } finally {
    unsubscribe();
  }
  expect(traces).toHaveLength(1);
  expect(traces[0].rootSpans[0].children[0].output).toBe(modelOutput);
  return traces[0];
}

afterAll(() => shutdownConfidentTrace());

describe("owned trace input and output", () => {
  const recordedCases = [
    ["missing fields", {}, requestInput, modelOutput],
    ["recorded null", { input: null, output: null }, null, null],
    ["recorded empty strings", { input: "", output: "" }, "", ""],
    ["recorded values", { input: 0, output: false }, 0, false],
    ["empty output array", { output: [] }, requestInput, modelOutput],
  ] as const;

  it.each(recordedCases)(
    "collects finished Mastra events with %s",
    async (_label, fields, input, output) => {
      const exporter = new DeepEvalExporter();
      const traceId = randomUUID().replaceAll("-", "");
      const rootId = "0011223344556677";
      const childId = "1122334455667788";
      try {
        const trace = await captureTrace(() =>
          traceContext(fields, async () => {
            await exporter.exportTracingEvent({
              type: "span_ended",
              exportedSpan: {
                id: childId,
                parentSpanId: rootId,
                traceId,
                name: "model",
                type: "model_inference",
                startTime: new Date(2000),
                endTime: new Date(3000),
                isRootSpan: false,
                isEvent: false,
                input: modelInput,
                output: modelOutput,
              },
            });
            await exporter.exportTracingEvent({
              type: "span_ended",
              exportedSpan: {
                id: rootId,
                traceId,
                name: "agent",
                type: "agent_run",
                startTime: new Date(1000),
                endTime: new Date(4000),
                isRootSpan: true,
                isEvent: false,
                input: requestInput,
              },
            });
            await exporter.flush();
          }),
        );
        expect(trace).toMatchObject({ input, output });
        expect(traceManager.createTraceApi(trace)).toMatchObject({
          input,
          output,
        });
      } finally {
        await exporter.shutdown();
      }
    },
  );

  it.each([
    [
      "local input",
      { input: "recorded input", output: "" },
      { input: null },
      { input: null, output: "" },
    ],
    [
      "local output",
      { input: null, output: "recorded output" },
      { output: "" },
      { input: null, output: "" },
    ],
  ] as const)(
    "keeps %s over recorded values and fills its sibling from the recorded trace",
    async (_label, recorded, update, expected) => {
      const trace = await captureTrace(() =>
        traceContext(recorded, () =>
          withSpan({ name: "agent", type: "agent" }, () => {
            updateCurrentTrace(update);
            withSpan({ name: "model", type: "llm" }, (span) =>
              span.setAttributes({
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": JSON.stringify([
                  {
                    role: "user",
                    parts: [{ type: "text", content: modelInput }],
                  },
                ]),
                "gen_ai.output.messages": JSON.stringify([
                  {
                    role: "assistant",
                    parts: [{ type: "text", content: modelOutput }],
                  },
                ]),
              }),
            );
          }),
        ),
      );
      expect(trace).toMatchObject(expected);
    },
  );
});
