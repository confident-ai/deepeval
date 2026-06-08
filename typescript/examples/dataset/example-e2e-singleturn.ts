import {
  EvaluationDataset,
  LLMTestCase,
  ToolCall,
  Golden,
  evaluate,
} from "../../src";

async function main() {
  const dataset = new EvaluationDataset();
  await dataset.pull({ alias: "asdf", finalized: true });

  for (const golden of dataset.goldens as Golden[]) {
    const testCase = new LLMTestCase({
      input: golden.input,
      actualOutput: "Hello",
      expectedOutput: "Hello",
      context: ["Hello"],
      retrievalContext: ["Hello"],
      toolsCalled: [
        new ToolCall({
          name: "tool",
          description: "tool",
          reasoning: "tool",
          output: "tool",
          inputParameters: {},
        }),
      ],
      expectedTools: [],
      reasoning: "tool",
      tokenCost: 0,
      completionTime: 0,
      name: "tool",
    });
    dataset.addTestCase(testCase);
  }

  await evaluate({
    testCases: dataset.testCases as LLMTestCase[],
    metricCollection: "New Collection",
  });
}

main().catch(console.error);
