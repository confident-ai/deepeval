import { EvaluationDataset, Golden } from "../../src";

const goldens = [new Golden({ input: "How tall is Mr. Everest?" })];
const dataset = new EvaluationDataset();
dataset.queue({ alias: "YOUR-DATASET-ALIAS", goldens });
