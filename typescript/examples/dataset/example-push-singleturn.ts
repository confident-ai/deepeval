import { EvaluationDataset, Golden } from "../../src";

const goldens = [new Golden({ input: "How tall is Mr. Everest?" })];
const dataset = new EvaluationDataset({ goldens: goldens });
dataset.push({ alias: "YOUR-DATASET-ALIAS", overwrite: true });
