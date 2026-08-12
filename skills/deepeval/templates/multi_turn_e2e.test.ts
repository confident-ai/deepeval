import { expect, it } from "vitest";
import { ConversationSimulator } from "deepeval";
import { ConversationalGolden, EvaluationDataset } from "deepeval/dataset";
import "deepeval/vitest";

import { MULTI_TURN_METRICS } from "./metrics";
// PLACEHOLDER: import the real chatbot model callback.
import { chatbotCallback } from "./ai-app";

const MAX_TURNS = 10;

const dataset = new EvaluationDataset();
await dataset.addGoldensFromJSON({ filePath: "tests/evals/.dataset.json" });

const simulator = new ConversationSimulator({ modelCallback: chatbotCallback });
const testCases = await simulator.simulate({
  conversationalGoldens: dataset.goldens as ConversationalGolden[],
  maxUserSimulations: MAX_TURNS,
});

it.each(testCases)("multi-turn e2e: $scenario", async (testCase) => {
  await expect(testCase).toPass(MULTI_TURN_METRICS);
});
