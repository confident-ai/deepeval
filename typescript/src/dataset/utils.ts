import { Golden, ConversationalGolden } from "./golden";
import { LLMTestCase, ConversationalTestCase, ToolCall } from "../test-case";
import { Turn } from "../test-case";

export function convertTestCasesToGoldens(testCases: LLMTestCase[]): Golden[] {
  const goldens: Golden[] = [];
  for (const testCase of testCases) {
    const llmTestCase = testCase as LLMTestCase;
    goldens.push(
      new Golden({
        input: llmTestCase.input,
        actualOutput: llmTestCase.actualOutput,
        expectedOutput: llmTestCase.expectedOutput,
        context: llmTestCase.context,
        retrievalContext: llmTestCase.retrievalContext,
        toolsCalled: llmTestCase.toolsCalled,
        expectedTools: llmTestCase.expectedTools,
        additionalMetadata: llmTestCase.additionalMetadata,
      }),
    );
  }
  return goldens;
}

export function convertGoldensToTestCases(
  goldens: Golden[],
  alias?: string,
  datasetId?: string,
): LLMTestCase[] {
  return goldens.map((golden, index) => {
    return createLLMTestCase({
      input: golden.input,
      actualOutput: golden.actualOutput || "",
      expectedOutput: golden.expectedOutput,
      context: golden.context,
      retrievalContext: golden.retrievalContext,
      additionalMetadata: golden.additionalMetadata,
      toolsCalled: golden.toolsCalled,
      expectedTools: golden.expectedTools,
      _datasetRank: index,
      _datasetAlias: alias,
      _datasetId: datasetId,
    });
  });
}

export function convertConvoGoldensToConvoTestCases(
  goldens: ConversationalGolden[],
  alias?: string,
  datasetId?: string,
): ConversationalTestCase[] {
  return goldens.map((golden, index) => {
    return createConversationalTestCase({
      turns: golden.turns,
      scenario: golden.scenario,
      userDescription: golden.userDescription,
      expectedOutcome: golden.expectedOutcome,
      context: golden.context,
      name: golden.name,
      additionalMetadata: golden.additionalMetadata,
      comments: golden.comments,
      _datasetRank: index,
      _datasetAlias: alias,
      _datasetId: datasetId,
    });
  });
}

function createLLMTestCase(params: {
  input: string;
  actualOutput: string;
  expectedOutput?: string;
  context?: string[];
  retrievalContext?: string[];
  additionalMetadata?: Record<string, any>;
  toolsCalled?: ToolCall[];
  expectedTools?: ToolCall[];
  comments?: string;
  reasoning?: string;
  tokenCost?: number;
  completionTime?: number;
  name?: string;
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;
}): LLMTestCase {
  return new LLMTestCase(params);
}

function createConversationalTestCase(params: {
  turns?: Turn[];
  chatbotRole?: string;
  scenario?: string;
  userDescription?: string;
  expectedOutcome?: string;
  context?: string[];
  name?: string;
  additionalMetadata?: Record<string, any>;
  comments?: string;
  tags?: string[];
  _datasetRank?: number;
  _datasetAlias?: string;
  _datasetId?: string;
}): ConversationalTestCase {
  return new ConversationalTestCase({
    ...params,
    turns: params.turns || [],
  });
}

export function trimAndLoadJson(jsonString: string): any {
  try {
    return JSON.parse(jsonString);
  } catch (_error) {
    const cleanedString = jsonString
      .replace(/^[\s\uFEFF\xA0\n\r]+|[\s\uFEFF\xA0\n\r]+$/g, "")
      .replace(/\\'/g, "'")
      .replace(/\\"/g, '"');
    try {
      return JSON.parse(cleanedString);
    } catch (innerError) {
      throw new Error(`Failed to parse JSON: ${innerError}`);
    }
  }
}

export function stripPrivateFields(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(stripPrivateFields);
  } else if (obj && typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj)
        .filter(([key]) => !key.startsWith("_")) // drop private fields
        .map(([key, value]) => [key, stripPrivateFields(value)]),
    );
  }
  return obj;
}

export const parseDelimited = (
  str: string | null | undefined,
  delimiter = ";",
): string[] => {
  if (!str) return [];
  return str
    .split(delimiter)
    .map((s) => s.trim())
    .filter(Boolean);
};

export const safeJsonParse = <T>(
  text: string | null | undefined,
  fallback: T,
): T => {
  try {
    return text ? JSON.parse(text) : fallback;
  } catch {
    return fallback;
  }
};
