import { Prompt } from "@/prompt";
import { setTracingContext } from "@/tracing/trace-context";
import { MODEL, getOpenRouterClient } from "./openrouter-clients";

const testPrompt = new Prompt({ alias: "openrouter-metric-collection-test" });
testPrompt.version = "01.00.00";
testPrompt.label = "production";
testPrompt.hash = "bab04ec";

/**
 * Stages a prompt, a metric collection and the extra evaluation params on the
 * LLM span, so the fixture pins that they reach the span.
 */
export async function invokeMetricCollectionApp(prompt: string) {
  const client = getOpenRouterClient();

  return await setTracingContext(
    {
      llmSpanContext: {
        prompt: testPrompt,
        metricCollection: "llm-span-evals",
        retrievalContext: ["Paris is the capital of France."],
        expectedOutput: "Paris",
      },
    },
    () =>
      client.chat.send({
        chatRequest: {
          model: MODEL,
          messages: [{ role: "user", content: prompt }],
        },
      }),
  );
}
