import { EvaluationDataset, ConversationalGolden } from "../../src";

const dataset = new EvaluationDataset();
const goldens = [
  new ConversationalGolden({ scenario: "User asking for refund." }),
];
dataset.queue({ alias: "YOUR-MULTITURN-ALIAS", goldens });
