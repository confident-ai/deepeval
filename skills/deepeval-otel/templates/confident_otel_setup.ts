/**
 * Raw OpenTelemetry -> Confident AI Observatory: minimal setup + example trace.
 *
 * Requires:
 *     npm install @opentelemetry/api @opentelemetry/sdk-trace-node @opentelemetry/exporter-trace-otlp-proto
 *     export CONFIDENT_API_KEY="<your Confident AI API key>"
 *
 * This template wires an OTLP/HTTP span exporter to Confident AI and emits one
 * example trace that demonstrates the `confident.*` attribute and data-type
 * contract. Run it directly to smoke-test the connection:
 *
 *     npx tsx confident_otel_setup.ts
 *
 * PLACEHOLDER: replace the example span/trace attribute values below with
 * values from the real application before using this as production
 * instrumentation.
 */

import { SpanStatusCode, trace, type Tracer } from "@opentelemetry/api";
// The OTLP/HTTP (protobuf) exporter. Confident AI accepts OTLP/HTTP only --
// never gRPC -- so do not use @opentelemetry/exporter-trace-otlp-grpc.
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import {
  BatchSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";

/**
 * Select the Confident AI OTLP endpoint from the API key region prefix.
 *
 * Only `confident_eu_...` keys use the EU endpoint; every other key
 * (`confident_us_...`, `confident_au_...`, or anything else) uses the
 * default.
 */
const pickEndpoint = (apiKey: string): string => {
  if (apiKey.startsWith("confident_eu_")) {
    return "https://eu.otel.confident-ai.com";
  }
  return "https://otel.confident-ai.com";
};

let provider: NodeTracerProvider;

/** Wire a NodeTracerProvider that exports to Confident AI over OTLP/HTTP. */
const configureTracing = (): Tracer => {
  const apiKey = process.env.CONFIDENT_API_KEY;
  if (!apiKey) {
    console.error(
      "CONFIDENT_API_KEY is not set. Export it before running:\n" +
        '    export CONFIDENT_API_KEY="<your Confident AI API key>"',
    );
    process.exit(1);
  }

  const endpoint = pickEndpoint(apiKey);
  provider = new NodeTracerProvider({
    spanProcessors: [
      new BatchSpanProcessor(
        new OTLPTraceExporter({
          // The exporter URL MUST include the /v1/traces suffix.
          url: `${endpoint}/v1/traces`,
          headers: { "x-confident-api-key": apiKey },
        }),
      ),
    ],
  });
  provider.register();
  return trace.getTracer("confident-otel-setup");
};

/** Emit one example trace: an agent span wrapping a child LLM span. */
const runExample = async (tracer: Tracer): Promise<void> => {
  // Root span. Trace-level attributes (confident.trace.*) can be set on any
  // span; the root is the natural place. Child spans nest automatically
  // because they open inside `startActiveSpan` (native OTel span context).
  await tracer.startActiveSpan("support-agent", async (root) => {
    root.setAttribute("confident.span.type", "agent");
    root.setAttribute("confident.agent.name", "support-agent");
    root.setAttribute("confident.span.input", "Where is my order?");

    // Trace-level attributes.
    root.setAttribute("confident.trace.name", "support-chat");
    root.setAttribute("confident.trace.input", "Where is my order?");
    // String lists are native OTLP arrays.
    root.setAttribute("confident.trace.tags", ["support", "example"]);
    // Dicts/metadata MUST be JSON-encoded strings (OTLP has no map type).
    root.setAttribute(
      "confident.trace.metadata",
      JSON.stringify({ app_version: "1.0.0", route: "order_status" }),
    );

    // Child LLM span.
    const answer = await tracer.startActiveSpan(
      "chat-completion",
      async (llm) => {
        llm.setAttribute("confident.span.type", "llm");
        llm.setAttribute("confident.llm.model", "gpt-4o");
        llm.setAttribute("confident.llm.input_token_count", 42);
        llm.setAttribute("confident.llm.output_token_count", 18);
        llm.setAttribute(
          "confident.span.metadata",
          JSON.stringify({ temperature: 0.2 }),
        );

        try {
          const output = "Your order ships tomorrow.";
          llm.setAttribute("confident.span.output", output);
          return output;
        } catch (err) {
          // Span errors use native OTel status, not a confident.* attribute.
          llm.setStatus({
            code: SpanStatusCode.ERROR,
            message: (err as Error).message,
          });
          llm.recordException(err as Error);
          throw err;
        } finally {
          llm.end();
        }
      },
    );

    root.setAttribute("confident.span.output", answer);
    root.setAttribute("confident.trace.output", answer);
    root.end();
  });
};

const main = async (): Promise<void> => {
  const tracer = configureTracing();
  await runExample(tracer);
  // Flush so the batch processor exports before the process exits.
  await provider.shutdown();
  console.log("Trace exported. Check the Confident AI Observatory.");
};

main();
