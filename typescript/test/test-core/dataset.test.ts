import * as fs from "fs";
import * as path from "path";

import { config } from "dotenv";

import { EvaluationDataset } from "@/dataset";
import { LLMTestCase } from "@/test-case";

config();

describe("Dataset Module", () => {
  // Create a temp CSV file for testing
  const createTempCsvFile = async () => {
    const tempDir = path.join(__dirname, "temp");
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir);
    }

    const csvPath = path.join(tempDir, "test_dataset.csv");
    const csvContent = `input,actual_output,expected_output,context,retrieval_context
"What is the capital of Germany?","Berlin is the capital of Germany.","Berlin","Geography;Europe","Germany is a country in Central Europe."
"What is the formula for water?","The chemical formula for water is H2O.","H2O","Chemistry;Molecules","Water is a chemical compound consisting of hydrogen and oxygen atoms."`;

    fs.writeFileSync(csvPath, csvContent);
    return csvPath;
  };

  // Clean up temp files after tests
  afterAll(() => {
    const tempDir = path.join(__dirname, "temp");
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test("Should create an empty dataset", () => {
    const dataset = new EvaluationDataset();
    expect(dataset.goldens.length).toBe(0);
  });

  test("Should add test cases to dataset", () => {
    const dataset = new EvaluationDataset();
    const testCase1 = new LLMTestCase({
      input: "Test input 1",
      actualOutput: "Test actual output 1",
      expectedOutput: "Test expected output 1",
    });
    dataset.addTestCase(testCase1);
    const testCase2 = new LLMTestCase({
      input: "Test input 2",
      actualOutput: "Test actual output 2",
      expectedOutput: "Test expected output 2",
    });
    dataset.addTestCase(testCase2);

    expect(dataset.testCases.length).toBe(2);
    const testCases = dataset.testCases;
    if (
      testCases[0] instanceof LLMTestCase &&
      testCases[1] instanceof LLMTestCase
    ) {
      expect(testCases[0].input).toBe("Test input 1");
      expect(testCases[1].input).toBe("Test input 2");
    }
  });

  test("Should load test cases from CSV file", async () => {
    const csvPath = await createTempCsvFile();
    const dataset = new EvaluationDataset();
    const testCases = await dataset.addTestCasesFromCSV({
      filePath: csvPath,
      contextDelimiter: ";",
    });
    const firstTestCase = testCases[0];

    expect(testCases.length).toBe(2);
    expect(dataset.testCases.length).toBe(2);
    if (firstTestCase instanceof LLMTestCase) {
      expect(firstTestCase.input).toBe("What is the capital of Germany?");
      expect(firstTestCase.actualOutput).toBe(
        "Berlin is the capital of Germany.",
      );
      expect(firstTestCase.expectedOutput).toBe("Berlin");
      expect(firstTestCase.context).toEqual(["Geography", "Europe"]);
      expect(firstTestCase.retrievalContext).toEqual([
        "Germany is a country in Central Europe.",
      ]);
    }
  });

  test("Should iterate through test cases", () => {
    const dataset = new EvaluationDataset();
    const testCase1 = new LLMTestCase({
      input: "Test input 1",
      actualOutput: "Test actual output 1",
    });
    dataset.addTestCase(testCase1);
    const testCase2 = new LLMTestCase({
      input: "Test input 2",
      actualOutput: "Test actual output 2",
    });
    dataset.addTestCase(testCase2);
    const inputs: string[] = [];
    for (const testCase of dataset.testCases) {
      if (testCase instanceof LLMTestCase) {
        inputs.push(testCase.input);
      }
    }

    expect(inputs).toEqual(["Test input 1", "Test input 2"]);
  });
});
