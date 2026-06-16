import { ConversationalTestCase, MultiTurnParams, Turn } from "../test-case";
import { DeepEvalError, MissingTestCaseParamsError } from "../errors";
import { BaseConversationalMetric } from "./base-conversational-metric";

/**
 * Verify a conversational test case provides the params a metric requires.
 * Mirrors Python's `check_conversational_test_case_params`. Sets `metric.error`
 * and throws `MissingTestCaseParamsError` (skippable by the runner).
 */
export function checkConversationalTestCaseParams(
  testCase: ConversationalTestCase,
  requiredParams: MultiTurnParams[],
  metric: BaseConversationalMetric,
  options: { requireChatbotRole?: boolean } = {},
): void {
  if (!(testCase instanceof ConversationalTestCase)) {
    const err = `Unable to evaluate test cases that are not of type 'ConversationalTestCase' using the conversational '${metric.name}' metric.`;
    metric.error = err;
    throw new DeepEvalError(err);
  }

  const fail = (msg: string): never => {
    metric.error = msg;
    throw new MissingTestCaseParamsError(msg);
  };

  if (
    requiredParams.includes(MultiTurnParams.EXPECTED_OUTCOME) &&
    testCase.expectedOutcome == null
  ) {
    fail(
      `'expected_outcome' in a conversational test case cannot be empty for the '${metric.name}' metric.`,
    );
  }
  if (
    requiredParams.includes(MultiTurnParams.SCENARIO) &&
    testCase.scenario == null
  ) {
    fail(
      `'scenario' in a conversational test case cannot be empty for the '${metric.name}' metric.`,
    );
  }
  if (
    requiredParams.includes(MultiTurnParams.METADATA) &&
    testCase.additionalMetadata == null
  ) {
    fail(
      `'metadata' in a conversational test case cannot be empty for the '${metric.name}' metric.`,
    );
  }
  if (requiredParams.includes(MultiTurnParams.TAGS) && testCase.tags == null) {
    fail(
      `'tags' in a conversational test case cannot be empty for the '${metric.name}' metric.`,
    );
  }
  if (options.requireChatbotRole && testCase.chatbotRole == null) {
    fail(
      `'chatbot_role' in a conversational test case cannot be empty for the '${metric.name}' metric.`,
    );
  }
  if (!testCase.turns || testCase.turns.length === 0) {
    fail("'turns' in conversational test case cannot be empty.");
  }
}

/**
 * Convert a `Turn` to a plain dict for prompt rendering (mirrors Python's
 * `convert_turn_to_dict`). Defaults to `{ role, content }`; conversation-level
 * params (scenario/expected_outcome/metadata/tags) are skipped. The dict KEYS
 * are the snake_case names the templates expect (kept independent of the
 * camelCase enum values).
 */
export function convertTurnToDict(
  turn: Turn,
  turnParams: MultiTurnParams[] = [
    MultiTurnParams.CONTENT,
    MultiTurnParams.ROLE,
  ],
): Record<string, unknown> {
  const skip = new Set<MultiTurnParams>([
    MultiTurnParams.SCENARIO,
    MultiTurnParams.EXPECTED_OUTCOME,
    MultiTurnParams.METADATA,
    MultiTurnParams.TAGS,
  ]);
  const fields: Partial<
    Record<MultiTurnParams, { key: string; get: () => unknown }>
  > = {
    [MultiTurnParams.ROLE]: { key: "role", get: () => turn.role },
    [MultiTurnParams.CONTENT]: { key: "content", get: () => turn.content },
    [MultiTurnParams.RETRIEVAL_CONTEXT]: {
      key: "retrieval_context",
      get: () => turn.retrievalContext,
    },
    [MultiTurnParams.TOOLS_CALLED]: {
      key: "tools_called",
      get: () => turn.toolsCalled,
    },
  };
  const result: Record<string, unknown> = {};
  for (const param of turnParams) {
    if (skip.has(param)) continue;
    const field = fields[param];
    if (!field) continue;
    const value = field.get();
    if (value != null) result[field.key] = value;
  }
  return result;
}

/**
 * Sliding windows over a list: window `i` is `items[max(0, i-size+1) .. i]`.
 * Generic so it works on turns OR on grouped unit-interactions (Python's
 * `get_turns_in_sliding_window`).
 */
export function getTurnsInSlidingWindow<T>(
  items: T[],
  windowSize: number,
): T[][] {
  const windows: T[][] = [];
  for (let i = 0; i < items.length; i++) {
    windows.push(items.slice(Math.max(0, i - windowSize + 1), i + 1));
  }
  return windows;
}

/**
 * Group turns into "unit interactions" — each a user→…→assistant block. A new
 * unit starts when a user turn follows an assistant turn (Python's
 * `get_unit_interactions`).
 */
export function getUnitInteractions(turns: Turn[]): Turn[][] {
  const units: Turn[][] = [];
  let current: Turn[] = [];
  let hasUser = false;

  for (const turn of turns) {
    if (
      current.length > 0 &&
      current[current.length - 1].role === "assistant" &&
      turn.role === "user" &&
      hasUser
    ) {
      units.push(current);
      current = [turn];
      hasUser = true;
      continue;
    }
    current.push(turn);
    if (turn.role === "user") hasUser = true;
  }

  if (
    current.length > 1 &&
    current[current.length - 1].role === "assistant" &&
    hasUser
  ) {
    units.push(current);
  }
  return units;
}
