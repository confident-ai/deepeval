import { EvaluationDataset, ConversationalGolden } from "../../src";

const goldens = [
  new ConversationalGolden({ scenario: "Angry user asking for a refund." }),
];
const dataset = new EvaluationDataset({ goldens });
dataset.push({ alias: "YOUR-MULTITURN-ALIAS" });
