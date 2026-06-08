import * as fs from "fs";
import * as path from "path";

import { config } from "dotenv";

import { EvaluationDataset, Golden } from "../../src/dataset";
import { LLMTestCase } from "../../src/test-case";

// process.env.CONFIDENT_API_KEY = "YOUR API KEY";
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
    const result = await dataset.addTestCasesFromCSV({
      filePath: csvPath,
      inputCol: "input",
      actualOutputCol: "actual_output",
      expectedOutputCol: "expected_output",
      contextCol: "context",
      contextDelimiter: ";",
      retrievalContextCol: "retrieval_context",
    });
    const testCases = result;
    const firstTestCase = testCases[0];

    expect(testCases.length).toBe(2);
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

  test("Should push dataset to Confident AI", async () => {
    const dataset = new EvaluationDataset();
    const goldenInput = new Golden({
      input: "This is an input, of a golden, for a QA Dataset",
    });
    dataset.addGolden(goldenInput);
    await dataset.push({
      alias: "QA Dataset",
    });
  });

  test("Should pull dataset from Confident AI", async () => {
    const dataset = new EvaluationDataset();
    await dataset.pull({
      alias: "QA Dataset",
    });

    expect(dataset.goldens.length).toBeGreaterThan(0);
  });

  test("Should create, list, and pull dataset versions", async () => {
    const versionAlias = "QA Dataset Versioning";

    const seed = new EvaluationDataset();
    seed.addGolden(
      new Golden({
        input: "Versioned input",
        expectedOutput: "Versioned expected",
      }),
    );
    await seed.push({ alias: versionAlias });

    const versionResult = await seed.createVersion({ alias: versionAlias });
    expect(typeof versionResult.version).toBe("string");
    expect(versionResult.version.length).toBeGreaterThan(0);

    const versions = await seed.getVersions({ alias: versionAlias });
    expect(versions.some((v) => v.version === versionResult.version)).toBe(
      true,
    );

    const pulled = new EvaluationDataset();
    await pulled.pull({
      alias: versionAlias,
      version: versionResult.version,
    });
    expect(pulled.goldens.length).toBeGreaterThan(0);

    const pulledLatest = new EvaluationDataset();
    await pulledLatest.pull({ alias: versionAlias });
    expect(pulledLatest.goldens.length).toBeGreaterThan(0);

    const followUp = new EvaluationDataset();
    followUp.addGolden(
      new Golden({
        input: "Follow-up input",
        expectedOutput: "Follow-up expected",
      }),
    );
    await followUp.push({
      alias: versionAlias,
      version: versionResult.version,
    });
  });
});
