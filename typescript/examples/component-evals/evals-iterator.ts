/**
 * Component-level evals over a dataset — score individual spans, locally.
 *
 *   npx ts-node examples/component-evals/evals-iterator.ts
 *
 * Needs OPENAI_API_KEY. CONFIDENT_API_KEY is optional: results print either way,
 * and with a key the run is also posted.
 */
import { metrics } from "deepeval";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { nextLlmSpan } from "deepeval/tracing";

import { ask, mastra } from "./weather-agent";

async function main() {
  const dataset = new EvaluationDataset({
    goldens: [
      new Golden({ input: "What's the weather in Tokyo?" }),
      new Golden({ input: "What's the weather in London?" }),
    ],
  });

  for await (const golden of dataset.evalsIterator({
    // TRACE-level: judges the whole turn against the golden.
    metrics: [new metrics.TaskCompletionMetric()],
  })) {
    await nextLlmSpan({ metrics: [new metrics.AnswerRelevancyMetric()] }, () =>
      ask((golden as Golden).input),
    );
  }

  // One row per golden, plus one per evaluated component (named after the span).
  for (const r of dataset.evalResults) {
    const scores = (r.metricsData ?? []).map((m) => `${m.name}=${m.score}`);
    console.log(`${r.name}: ${scores.join(", ")}`);
  }

  await mastra.observability.shutdown();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

/* ---------------------------------------------------------------------------
 * The same two seams in the other integrations. Only the framework call changes;
 * `updateCurrentSpan` / `next*Span` are identical everywhere.
 *
 *   LangChain      await nextLlmSpan({ metrics: [...] }, () =>
 *                    chain.invoke(input, { callbacks: [handler] }));
 *
 *   AI SDK         const tracer = configureAiSdkTracing({ name: "app" });
 *                  await nextLlmSpan({ metrics: [...] }, () => generateText({
 *                    model, prompt, experimental_telemetry: { isEnabled: true, tracer },
 *                  }));
 *
 *   OpenAI Agents  setTraceProcessors([new DeepEvalTracingProcessor()]);
 *                  await nextLlmSpan({ metrics: [...] }, () => run(agent, input));
 *                  // register FIRST if you also want `updateCurrentSpan` in tools
 *
 *   OpenInference  instrumentOpenInference({ name: "app" });
 *                  await nextLlmSpan({ metrics: [...] }, () => myInstrumentedApp(input));
 *
 *   OpenAI client  instrumentOpenAI(client);
 *                  await nextLlmSpan({ metrics: [...] }, () => client.chat.completions.create(...));
 *
 *   Plain observe  observe({ type: SpanType.TOOL, metrics: [...], fn: myTool })
 *                  // metrics can be passed straight to `observe`
 *
 * Other span types: nextAgentSpan, nextToolSpan, nextRetrieverSpan, nextSpan.
 *
 * Scope-wide instead of one-shot — applies to EVERY matching span in scope:
 *
 *   await setTracingContext(
 *     { llmSpanContext: { metrics: [...], toolsMetrics: [...] } },
 *     () => ask(input),
 *   );
 * ------------------------------------------------------------------------- */
