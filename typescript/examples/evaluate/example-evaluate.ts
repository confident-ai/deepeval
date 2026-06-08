import { LLMTestCase, ToolCall } from "../../src/test-case";
import { evaluate } from "../../src/confident/evaluate";

async function main() {
  const testCase1 = new LLMTestCase({
    input: "What is the capital of Germany?",
    actualOutput: "Berlin is the capital of Germany.",
    expectedOutput: "Berlin",
    context: ["Geography", "Europe"],
    retrievalContext: ["Germany is a country in Central Europe."],
  });
  const testCase2 = new LLMTestCase({
    input: "What is the formula for water?",
    actualOutput: "The chemical formula for water is H2O.",
    expectedOutput: "H2O",
    context: ["Chemistry", "Molecules"],
    retrievalContext: [
      "Water is a chemical compound consisting of hydrogen and oxygen atoms.",
    ],
  });
  const toolCall = new ToolCall({
    name: "search_web",
    description: "Search the web for information",
    reasoning: "Need to find information about water",
    output: { results: ["Water is H2O"] },
    inputParameters: { query: "chemical formula for water" },
  });
  const testCase3 = new LLMTestCase({
    input: "What is the chemical formula for water?",
    actualOutput: "The chemical formula for water is H2O.",
    expectedOutput: "H2O",
    context: ["Chemistry"],
    retrievalContext: ["Water is composed of hydrogen and oxygen"],
    additionalMetadata: { source: "chemistry textbook" },
    comments: "Example with tool calls",
    toolsCalled: [toolCall],
  });
  const testCases = [testCase1, testCase2, testCase3];

  try {
    const metricCollection = "New Collection";
    await evaluate({
      metricCollection,
      llmTestCases: testCases,
    });
  } catch (error: any) {
    console.error("Error evaluating test cases:", error);
  }
}

main().catch((error) => {
  console.error("Error running example:", error);
});
