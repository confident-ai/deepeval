import { ConversationalTestCase, LLMTestCase } from "../test-case";

export interface ConfidentEvaluateRequestData {
  metricCollection: string;
  llmTestCases?: LLMTestCase[];
  conversationalTestCases?: ConversationalTestCase[];
  hyperparameters?: { [key: string]: string };
  identifier?: string;
}

export interface ConfidentEvaluateResponseData {
  link: string;
}
