import { LLMTestCase, ToolCall } from "@/test-case";

export const JUDGE_EVAL_VARIABLE_PATTERN =
  /\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}/g;
export const JUDGE_EVAL_ALL_ITEMS = "*";

export enum JudgeEvalField {
  INPUT = "input",
  OUTPUT = "output",
  METADATA = "metadata",
  TOOLS_CALLED = "tools_called",
}

export enum JudgeEvalRole {
  SYSTEM = "system",
  USER = "user",
  ASSISTANT = "assistant",
}

export interface JudgeEvalMessage {
  role: JudgeEvalRole;
  content: string;
}

export type JudgeEvalPathSegment = string | number;

export interface JudgeEvalVariable {
  field: JudgeEvalField;
  path?: JudgeEvalPathSegment[];
}

export function extractVariableNames(messages: JudgeEvalMessage[]): string[] {
  const names: string[] = [];
  for (const message of messages) {
    for (const match of message.content.matchAll(JUDGE_EVAL_VARIABLE_PATTERN)) {
      if (!names.includes(match[1])) names.push(match[1]);
    }
  }
  return names;
}

export function validateMessages(
  messages: JudgeEvalMessage[] | undefined,
): JudgeEvalMessage[] {
  if (!messages || messages.length === 0) {
    throw new Error("JudgeEval needs at least one message.");
  }
  messages.forEach((message, index) => {
    if (message.role === JudgeEvalRole.SYSTEM && index !== 0) {
      throw new Error("A system message is only allowed as the first message.");
    }
  });
  return [...messages];
}

export function validateVariables(
  messages: JudgeEvalMessage[],
  variables: Record<string, JudgeEvalVariable> | undefined,
): Record<string, JudgeEvalVariable> {
  const mapped = { ...(variables ?? {}) };
  const unmapped = extractVariableNames(messages).filter(
    (name) => !(name in mapped),
  );
  if (unmapped.length > 0) {
    throw new Error(
      `Every variable in the messages needs a mapping. Missing: ${unmapped.join(", ")}`,
    );
  }
  return mapped;
}

export function validateScoreRange(
  scoreRange: [number, number],
): [number, number] {
  if (scoreRange.length !== 2) {
    throw new Error("scoreRange must be a [min, max] pair.");
  }
  const [minimum, maximum] = scoreRange;
  if (minimum >= maximum) {
    throw new Error("scoreRange min must be lower than max.");
  }
  return [minimum, maximum];
}

export function parseJson(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function serializeToolCall(toolCall: ToolCall): Record<string, unknown> {
  const serialized: Record<string, unknown> = {
    name: toolCall.name,
    description: toolCall.description,
    type: toolCall.type,
    reasoning: toolCall.reasoning,
    output: toolCall.output,
    input_parameters: toolCall.inputParameters,
  };
  return Object.fromEntries(
    Object.entries(serialized).filter(([, value]) => value != null),
  );
}

export function readField(
  testCase: LLMTestCase,
  field: JudgeEvalField,
): unknown {
  switch (field) {
    case JudgeEvalField.INPUT:
      return parseJson(testCase.input);
    case JudgeEvalField.OUTPUT:
      return parseJson(testCase.actualOutput);
    case JudgeEvalField.METADATA:
      return testCase.additionalMetadata;
    case JudgeEvalField.TOOLS_CALLED:
      return testCase.toolsCalled && testCase.toolsCalled.length > 0
        ? testCase.toolsCalled.map(serializeToolCall)
        : undefined;
  }
}

export function resolvePath(
  value: unknown,
  path: JudgeEvalPathSegment[],
): unknown {
  if (path.length === 0) return value;

  const current = parseJson(value);
  const [key, ...rest] = path;

  if (key === JUDGE_EVAL_ALL_ITEMS) {
    if (!Array.isArray(current)) return undefined;
    return current
      .map((item) => resolvePath(item, rest))
      .filter((match) => match !== undefined && match !== null);
  }

  if (typeof key === "number") {
    if (
      Array.isArray(current) &&
      Number.isInteger(key) &&
      key >= -current.length &&
      key < current.length
    ) {
      return resolvePath(current.at(key), rest);
    }
    return undefined;
  }

  if (
    current !== null &&
    typeof current === "object" &&
    !Array.isArray(current) &&
    Object.prototype.hasOwnProperty.call(current, key)
  ) {
    return resolvePath((current as Record<string, unknown>)[key], rest);
  }
  return undefined;
}

export function stringify(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export function resolveVariables(
  testCase: LLMTestCase,
  variables: Record<string, JudgeEvalVariable>,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(variables).map(([name, variable]) => [
      name,
      stringify(
        resolvePath(readField(testCase, variable.field), variable.path ?? []),
      ),
    ]),
  );
}

export function fillVariables(
  content: string,
  values: Record<string, string>,
): string {
  return content.replace(JUDGE_EVAL_VARIABLE_PATTERN, (match, name: string) =>
    name in values ? values[name] : match,
  );
}

export function renderMessages(
  messages: JudgeEvalMessage[],
  values: Record<string, string>,
): string {
  return messages
    .map(
      (message) =>
        `${message.role.charAt(0).toUpperCase()}${message.role.slice(1)}:\n${fillVariables(message.content, values)}`,
    )
    .join("\n\n");
}
