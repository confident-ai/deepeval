import { expect, it } from "vitest";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { LLMTestCase } from "deepeval/test-case";
import "deepeval/vitest";

import { SINGLE_TURN_NO_TRACING_METRICS } from "./metrics";
// PLACEHOLDER: import the real app entry point.
import { runAiApp } from "./ai-app";

const dataset = new EvaluationDataset();
await dataset.addGoldensFromJSON({ filePath: "tests/evals/.dataset.json" });

it.each(dataset.goldens as Golden[])(
  "single-turn no tracing: $input",
  async (golden) => {
    const actualOutput = await runAiApp(golden.input);
    const testCase = new LLMTestCase({
      input: golden.input,
      actualOutput,
      expectedOutput: golden.expectedOutput,
      context: golden.context,
      retrievalContext: golden.retrievalContext,
    });
    await expect(testCase).toPass(SINGLE_TURN_NO_TRACING_METRICS);
  },
);
