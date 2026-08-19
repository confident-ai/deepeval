import { Golden, ConversationalGolden } from "@/dataset/golden";
import {
  LLMTestCase,
  ConversationalTestCase,
  MCPPromptCall,
  MCPResourceCall,
  MCPToolCall,
  RetrievedContextData,
  ToolCall,
  ToolCallType,
} from "@/test-case";
import { Turn, resolveRetrievalContext } from "@/test-case";

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
        retrievalContext: resolveRetrievalContext(llmTestCase.retrievalContext),
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
      name: golden.name,
      comments: golden.comments,
      _datasetRank: index,
      _datasetAlias: alias,
      _datasetId: datasetId,
    });
  });
}

export function convertConvoTestCasesToConvoGoldens(
  testCases: ConversationalTestCase[],
): ConversationalGolden[] {
  return testCases.map((testCase) => {
    if (!testCase.scenario) {
      throw new Error(
        "Please provide a scenario in your 'ConversationalTestCase' to convert it to a 'ConversationalGolden'.",
      );
    }
    return new ConversationalGolden({
      scenario: testCase.scenario,
      turns: testCase.turns,
      expectedOutcome: testCase.expectedOutcome,
      userDescription: testCase.userDescription,
      context: testCase.context,
      additionalMetadata: testCase.additionalMetadata,
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
  retrievalContext?: (string | RetrievedContextData)[];
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

/** Keeps a `RetrievedContextData`'s source on disk. Matches Python's marker. */
const RETRIEVED_CONTEXT_MARKER =
  /^deepeval_source=([\s\S]*?),deepeval_context=([\s\S]*)$/;

export function serializeRetrievalContext(
  retrievalContext: (string | RetrievedContextData)[] | undefined,
): string[] | undefined {
  return retrievalContext?.map((item) =>
    item instanceof RetrievedContextData
      ? `deepeval_source=${item.source},deepeval_context=${item.context}`
      : item,
  );
}

/** For a csv or jsonl cell, which holds one string rather than a list. */
export function joinRetrievalContext(
  retrievalContext: (string | RetrievedContextData)[] | undefined,
  delimiter = "|",
): string | undefined {
  return serializeRetrievalContext(retrievalContext)?.join(delimiter);
}

/** Drops unset fields, as Python's `exclude_none` model dump does. */
export function serializeModels(
  models: object[] | undefined,
): Record<string, unknown>[] | undefined {
  if (!models || models.length === 0) return undefined;
  return models.map((model) =>
    Object.fromEntries(
      Object.entries(model).filter(([, value]) => value != null),
    ),
  );
}

export function formatTurns(turns: Turn[]): string {
  return JSON.stringify(
    turns.map((turn) => ({
      role: turn.role,
      content: turn.content,
      user_id: turn.userId ?? null,
      retrieval_context:
        serializeRetrievalContext(turn.retrievalContext) ?? null,
      tools_called: serializeModels(turn.toolsCalled) ?? null,
      mcp_tools_called: serializeModels(turn.mcpToolsCalled) ?? null,
      mcp_resources_called: serializeModels(turn.mcpResourcesCalled) ?? null,
      mcp_prompts_called: serializeModels(turn.mcpPromptsCalled) ?? null,
      metadata: turn.additionalMetadata ?? null,
    })),
  );
}

export function reconstructRetrievalContext(
  retrievalContext: unknown,
): (string | RetrievedContextData)[] | undefined {
  if (retrievalContext == null) return undefined;
  if (!Array.isArray(retrievalContext)) {
    throw new TypeError(
      "Expected 'retrievalContext' to be an array, a delimited string, or null.",
    );
  }
  return retrievalContext.map((item) => {
    if (typeof item !== "string") return item;
    const match = RETRIEVED_CONTEXT_MARKER.exec(item);
    return match
      ? new RetrievedContextData({ source: match[1], context: match[2] })
      : item;
  });
}

/** Read a key, falling back to its camelCase spelling. */
export function pickKey(record: Record<string, unknown>, key: string): unknown {
  const value = record[key];
  if (value !== undefined) return value;
  return record[key.replace(/_([a-z])/g, (_m, c: string) => c.toUpperCase())];
}

function parseStringList(
  value: unknown,
  delimiter: string,
): string[] | undefined {
  if (value == null) return undefined;
  if (Array.isArray(value)) return value as string[];
  if (typeof value === "string") return value ? value.split(delimiter) : [];
  throw new TypeError(
    "Expected a context field to be an array, a delimited string, or null.",
  );
}

export function parseToolCalls(value: unknown): ToolCall[] | undefined {
  if (value == null) return undefined;
  const raw = typeof value === "string" ? trimAndLoadJson(value) : value;
  if (!Array.isArray(raw)) {
    throw new TypeError("Expected a JSON string or an array of tool calls.");
  }
  if (raw.length === 0) return undefined;
  return raw.map((tool) => {
    if (tool instanceof ToolCall) return tool;
    if (tool == null || typeof tool !== "object") {
      throw new TypeError("Each tool call must be an object with a 'name'.");
    }
    const t = tool as Record<string, unknown>;
    if (typeof t.name !== "string") {
      throw new TypeError("Each tool call must have a string 'name'.");
    }
    return new ToolCall({
      name: t.name,
      description: t.description as string | undefined,
      type: t.type as ToolCallType | undefined,
      reasoning: t.reasoning as string | undefined,
      output: t.output,
      inputParameters: pickKey(t, "input_parameters") as
        | Record<string, any>
        | undefined,
    });
  });
}

function parseMcpCalls<T>(
  value: unknown,
  field: string,
  make: (raw: any) => T,
): T[] | undefined {
  if (value == null) return undefined;
  if (!Array.isArray(value)) {
    throw new TypeError(`Expected '${field}' to be an array.`);
  }
  return value.map(make);
}

export function parseTurns(value: unknown): Turn[] {
  const raw = typeof value === "string" ? trimAndLoadJson(value) : value;
  if (!Array.isArray(raw)) {
    throw new TypeError("Expected a JSON string or an array of turns.");
  }
  return raw.map((turn, index) => {
    if (turn instanceof Turn) return turn;
    if (turn == null || typeof turn !== "object") {
      throw new TypeError(`Turn at index ${index} is not an object.`);
    }
    const t = turn as Record<string, unknown>;
    if (t.role !== "user" && t.role !== "assistant") {
      throw new TypeError(`Turn at index ${index} is missing a valid 'role'.`);
    }
    if (typeof t.content !== "string") {
      throw new TypeError(
        `Turn at index ${index} is missing a valid 'content'.`,
      );
    }
    return new Turn({
      role: t.role,
      content: t.content,
      userId: pickKey(t, "user_id") as string | undefined,
      retrievalContext: reconstructRetrievalContext(
        pickKey(t, "retrieval_context"),
      ),
      toolsCalled: parseToolCalls(pickKey(t, "tools_called")),
      mcpToolsCalled: parseMcpCalls(
        pickKey(t, "mcp_tools_called"),
        "mcpToolsCalled",
        (c) => new MCPToolCall(c),
      ),
      mcpResourcesCalled: parseMcpCalls(
        pickKey(t, "mcp_resources_called"),
        "mcpResourcesCalled",
        (c) => new MCPResourceCall(c),
      ),
      mcpPromptsCalled: parseMcpCalls(
        pickKey(t, "mcp_prompts_called"),
        "mcpPromptsCalled",
        (c) => new MCPPromptCall(c),
      ),
      additionalMetadata: (t.metadata ?? pickKey(t, "additional_metadata")) as
        | Record<string, any>
        | undefined,
    });
  });
}

export interface GoldenKeyNames {
  input: string;
  actualOutput: string;
  expectedOutput: string;
  context: string;
  retrievalContext: string;
  toolsCalled: string;
  expectedTools: string;
  comments: string;
  name: string;
  sourceFile: string;
  additionalMetadata: string;
  customColumnKeyValues: string;
  scenario: string;
  turns: string;
  expectedOutcome: string;
  userDescription: string;
}

export const DEFAULT_GOLDEN_KEY_NAMES: GoldenKeyNames = {
  input: "input",
  actualOutput: "actual_output",
  expectedOutput: "expected_output",
  context: "context",
  retrievalContext: "retrieval_context",
  toolsCalled: "tools_called",
  expectedTools: "expected_tools",
  comments: "comments",
  name: "name",
  sourceFile: "source_file",
  additionalMetadata: "additional_metadata",
  customColumnKeyValues: "custom_column_key_values",
  scenario: "scenario",
  turns: "turns",
  expectedOutcome: "expected_outcome",
  userDescription: "user_description",
};

/** A record carrying a truthy `scenario` becomes a `ConversationalGolden`. */
export function goldenFromRecord(
  record: Record<string, unknown>,
  keys: GoldenKeyNames,
  delimiters: { context: string; retrievalContext: string },
): Golden | ConversationalGolden {
  const context = parseStringList(
    pickKey(record, keys.context),
    delimiters.context,
  );
  const comments = pickKey(record, keys.comments) as string | undefined;
  const name = pickKey(record, keys.name) as string | undefined;
  const additionalMetadata = pickKey(record, keys.additionalMetadata) as
    | Record<string, any>
    | undefined;
  const customColumnKeyValues = pickKey(record, keys.customColumnKeyValues) as
    | Record<string, string>
    | undefined;

  const scenario = pickKey(record, keys.scenario);
  if (scenario) {
    const turns = pickKey(record, keys.turns);
    return new ConversationalGolden({
      scenario: String(scenario),
      turns: turns ? parseTurns(turns) : [],
      expectedOutcome: pickKey(record, keys.expectedOutcome) as
        | string
        | undefined,
      userDescription: pickKey(record, keys.userDescription) as
        | string
        | undefined,
      context,
      comments,
      name,
      additionalMetadata,
      customColumnKeyValues,
    });
  }

  return new Golden({
    input: pickKey(record, keys.input) as string,
    actualOutput: pickKey(record, keys.actualOutput) as string | undefined,
    expectedOutput: pickKey(record, keys.expectedOutput) as string | undefined,
    context,
    retrievalContext: reconstructRetrievalContext(
      parseStringList(
        pickKey(record, keys.retrievalContext),
        delimiters.retrievalContext,
      ),
    ),
    toolsCalled: parseToolCalls(pickKey(record, keys.toolsCalled)),
    expectedTools: parseToolCalls(pickKey(record, keys.expectedTools)),
    additionalMetadata,
    customColumnKeyValues,
    comments,
    name,
    sourceFile: pickKey(record, keys.sourceFile) as string | undefined,
  });
}
