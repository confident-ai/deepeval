import { expect, it } from "vitest";
import { EvaluationDataset, Golden } from "deepeval/dataset";
import "deepeval/vitest";

import { SINGLE_TURN_TRACE_METRICS } from "./metrics";
// PLACEHOLDER: import the real traced app entry point.
import { runTracedAiApp } from "./ai-app";

const dataset = new EvaluationDataset();
await dataset.addGoldensFromJSON({ filePath: "tests/evals/.dataset.json" });

it.each(dataset.goldens as Golden[])(
  "single-turn tracing: $input",
  async (golden) => {
    await expect(golden).toPass(SINGLE_TURN_TRACE_METRICS, {
      task: (g) => runTracedAiApp(g.input),
    });
  },
);
