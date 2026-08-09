import { config } from "dotenv";

import { EvaluationDataset, Golden } from "@/dataset";

config();

describe("Dataset Confident AI", () => {
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
