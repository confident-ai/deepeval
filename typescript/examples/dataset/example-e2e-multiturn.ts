import { EvaluationDataset, evaluate } from "../../src";

async function main() {
  const dataset = new EvaluationDataset();
  await dataset.pull({
    alias: "test_multi_turn_realistic_push",
    finalized: true,
    autoConvertGoldensToTestCases: true,
  });
  await evaluate({
    testCases: dataset.testCases,
    metricCollection: "My online evals",
  });
}

main().catch(console.error);
