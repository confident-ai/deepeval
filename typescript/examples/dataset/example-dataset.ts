// /**
//  * Example script demonstrating how to work with datasets in deepeval.ts
//  *
//  * This script shows:
//  * 1. Loading a dataset from a CSV file
//  * 2. Creating a dataset programmatically
//  * 3. Accessing and iterating through test cases
//  * 4. Pushing a dataset to Confident AI
//  * 5. Pulling a dataset from Confident AI
//  */

// // For local development (using relative paths):
import { EvaluationDataset } from "../../src/dataset";



// // When using as an installed package, you can import in several ways:
// // Option 1: Import specific types directly from the root
// // import { EvaluationDataset, LLMTestCase, ToolCall } from "deepeval-ts";

// // Option 2: Import from specific submodules
// // import { EvaluationDataset } from "deepeval-ts/dataset";
// // import { LLMTestCase, ToolCall } from "deepeval-ts/testCase";

// async function main() {
//   console.log("DeepEval.ts Dataset Examples\n");

//   // Example 1: Loading a dataset from a CSV file
//   console.log("Example 1: Loading dataset from CSV file");
//   const csvDataset = new EvaluationDataset();

//   try {
//     // Path to the sample CSV file
//     const csvFilePath = path.resolve(
//       __dirname,
//       "../__tests__/dataset/sample_dataset.csv"
//     );

//     // Load test cases from the CSV file
//     await csvDataset.addTestCasesFromCsvFile({
//       filePath: csvFilePath,
//       inputColName: "input",
//       actualOutputColName: "actual_output",
//       expectedOutputColName: "expected_output",
//       contextColName: "context",
//       contextColDelimiter: ";",
//       retrievalContextColName: "retrieval_context",
//       retrievalContextColDelimiter: ";",
//     });

//     console.log(`Successfully loaded ${csvDataset.length} test cases from CSV`);
//     console.log("First test case:");
//     console.log(`- Input: ${csvDataset.testCases[0].input}`);
//     console.log(`- Actual Output: ${csvDataset.testCases[0].actualOutput}`);
//     console.log(`- Expected Output: ${csvDataset.testCases[0].expectedOutput}`);
//     console.log(`- Context: ${csvDataset.testCases[0].context?.join(", ")}`);
//     console.log(
//       `- Retrieval Context: ${csvDataset.testCases[0].retrievalContext?.join(
//         ", "
//       )}`
//     );
//     console.log(csvDataset);
//   } catch (error) {
//     console.error("Error loading CSV dataset:", error);
//   }

//   // Example 2: Creating a dataset programmatically
//   console.log("Example 2: Creating dataset programmatically");

//   // Create test cases
//   const testCase1 = new LLMTestCase({
//     input: "What is the capital of Germany?",
//     actualOutput: "Berlin",
//     expectedOutput: "Berlin",
//     context: ["Geography", "Europe"],
//     retrievalContext: ["Germany is a country in Central Europe."],
//   });

//   const testCase2 = new LLMTestCase({
//     input: "What is the formula for water?",
//     actualOutput: "H2O",
//     expectedOutput: "H2O",
//     context: ["Chemistry", "Molecules"],
//     retrievalContext: [
//       "Water is a chemical compound consisting of hydrogen and oxygen atoms.",
//     ],
//   });

//   // Create a tool call example
//   const toolCall = new ToolCall({
//     name: "search_web",
//     description: "Search the web for information",
//     reasoning: "Need to find information about water",
//     output: { results: ["Water is H2O"] },
//     inputParameters: { query: "chemical formula for water" },
//   });

//   const testCase3 = new LLMTestCase({
//     input: "What is the chemical formula for water?",
//     actualOutput: "The chemical formula for water is H2O",
//     expectedOutput: "H2O",
//     context: ["Chemistry"],
//     retrievalContext: ["Water is composed of hydrogen and oxygen"],
//     additionalMetadata: { source: "chemistry textbook" },
//     comments: "Example with tool calls",
//     toolsCalled: [toolCall],
//   });

//   // Create dataset with test cases
//   const programmaticDataset = new EvaluationDataset([
//     testCase1,
//     testCase2,
//     testCase3,
//   ]);

//   console.log(`Created dataset with ${programmaticDataset.length} test cases`);
//   console.log("Test case with tool calls:");
//   console.log(`- Input: ${programmaticDataset.testCases[2].input}`);
//   console.log(
//     `- Tool Called: ${programmaticDataset.testCases[2].toolsCalled?.[0].name}`
//   );
//   console.log(
//     `- Tool Parameters: ${JSON.stringify(
//       programmaticDataset.testCases[2].toolsCalled?.[0].inputParameters
//     )}`
//   );
//   console.log();

//   // Example 3: Iterating through test cases
//   console.log("Example 3: Iterating through test cases");
//   console.log("All test cases in the programmatic dataset:");

//   let index = 0;
//   for (const testCase of programmaticDataset) {
//     console.log(`[${index++}] ${testCase.input} -> ${testCase.actualOutput}`);
//   }

//   // Example 4: Push dataset to Confident AI
//   console.log("\nExample 4: Pushing dataset to Confident AI");
//   try {
//     // Create a new dataset for pushing
//     const pushDataset = new EvaluationDataset();

//     // Add some test cases
//     pushDataset.addTestCase(
//       new LLMTestCase({
//         input: "What is the capital of France?",
//         actualOutput: "Paris is the capital of France.",
//         expectedOutput: "Paris",
//         context: ["Geography", "Europe"],
//         retrievalContext: ["France is a country in Western Europe."],
//       })
//     );

//     pushDataset.addTestCase(
//       new LLMTestCase({
//         input: "What is the formula for water?",
//         actualOutput: "The chemical formula for water is H2O.",
//         expectedOutput: "H2O",
//         context: ["Chemistry", "Molecules"],
//         retrievalContext: [
//           "Water is a compound consisting of hydrogen and oxygen.",
//         ],
//       })
//     );

//     // Push the dataset with alias "DataWiz QA Dataset"
//     // Set overwrite to true to replace an existing dataset with the same alias
//     // Set autoConvertTestCasesToGoldens to true to convert test cases to goldens
//     await pushDataset.push({
//       alias: "DataWiz QA Dataset",
//       overwrite: true,
//       autoConvertTestCasesToGoldens: true,
//     });
//     console.log(
//       `Successfully pushed dataset with ${pushDataset.length} test cases`
//     );
//   } catch (error: any) {
//     console.error("Error pushing dataset:", error);
//     // If the error is about not being logged in, provide guidance
//     if (error.message && error.message.includes("deepeval login")) {
//       console.log(
//         "\nTo push datasets to Confident AI, you need to set your API key."
//       );
//       console.log(
//         "You can do this by setting the CONFIDENT_API_KEY environment variable:"
//       );
//       console.log("export CONFIDENT_API_KEY='your-api-key-here'");
//     }
//   }

//   // Example 5: Pull dataset from Confident AI
//   console.log("\nExample 5: Pulling dataset from Confident AI");
//   try {
//     // Create a new dataset instance for pulling
//     const pullDataset = new EvaluationDataset();

//     // Pull the dataset with alias "DataWiz QA Dataset"
//     await pullDataset.pull({
//       alias: "DataWiz QA Dataset",
//       autoConvertGoldensToTestCases: false,
//     });

//     // Display the first test case
//     if (pullDataset.length > 0) {
//       const firstTestCase = pullDataset.testCases[0] as LLMTestCase;
//       console.log("\nFirst test case from pulled dataset:");
//       console.log(`- Input: ${firstTestCase.input}`);
//       console.log(`- Actual Output: ${firstTestCase.actualOutput}`);
//       console.log(`- Expected Output: ${firstTestCase.expectedOutput}`);
//       if (firstTestCase.context && firstTestCase.context.length > 0) {
//         console.log(`- Context: ${firstTestCase.context.join(", ")}`);
//       }
//       if (
//         firstTestCase.retrievalContext &&
//         firstTestCase.retrievalContext.length > 0
//       ) {
//         console.log(
//           `- Retrieval Context: ${firstTestCase.retrievalContext.join(", ")}`
//         );
//       }
//     }
//   } catch (error: any) {
//     console.error("Error pulling dataset:", error);
//     // If the error is about not being logged in, provide guidance
//     if (error.message && error.message.includes("deepeval login")) {
//       console.log(
//         "\nTo pull datasets from Confident AI, you need to set your API key."
//       );
//       console.log(
//         "You can do this by setting the CONFIDENT_API_KEY environment variable:"
//       );
//       console.log("export CONFIDENT_API_KEY='your-api-key-here'");
//     }
//   }
// }

// // Run the main function
// main().catch((error) => {
//   console.error("Error running example:", error);
// });
process.env.CONFIDENT_API_KEY = "YOUR API KEY";
async function main() {
  const dataset = new EvaluationDataset();
  await dataset.pull({
    alias: "testset",
    autoConvertGoldensToTestCases: true,
  });

  console.log(dataset.goldens[0]);
  console.log(dataset.testCases[0]);
}

main();
