import { Golden, EvaluationDataset } from "../../src";

const golden = new Golden({
  input: "Scenario",
  customColumnKeyValues: { Key: "Value" },
});

const dataset = new EvaluationDataset();

dataset.addGolden(golden);

dataset.push({
  alias: "Singleturn dataset",
  overwrite: true,
});
