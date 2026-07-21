import { Agent } from "@mastra/core/agent";
import type { DeepEvalExporter } from "../../../../src/integrations/mastra";
import { buildMastra } from "./mastra-harness";

const simpleAgent = new Agent({
  id: "simple-agent",
  name: "Simple Agent",
  instructions: "You are a helpful assistant. Answer in one short sentence.",
  model: "openai/gpt-4o-mini",
});

export async function runSimpleApp(exporter: DeepEvalExporter, prompt: string) {
  const mastra = buildMastra(exporter, { agents: { simpleAgent } });
  return await mastra.getAgent("simpleAgent").generate(prompt);
}
