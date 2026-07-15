import {
  SingleTurnParams,
  ArenaTestCase,
  LLMTestCase,
  ToolCall,
  resolveRetrievalContext,
} from "../../test-case";

// Fake names for masking contestant identities before judging (avoids name bias).
const FAKE_NAMES = [
  "Alice",
  "Bob",
  "Charlie",
  "Diana",
  "Eve",
  "Frank",
  "Grace",
  "Henry",
  "Iris",
  "Jack",
];

/** Fisher–Yates shuffle (Math.random is fine in runtime code). */
function shuffled<T>(items: T[]): T[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** A single contestant's output, limited to the evaluation params, as a JSON string. */
function formattedLLMTestCase(
  params: SingleTurnParams[],
  tc: LLMTestCase,
): string {
  const data: Record<string, unknown> = {};
  if (params.includes(SingleTurnParams.ACTUAL_OUTPUT) && tc.actualOutput != null)
    data.actual_output = tc.actualOutput;
  if (params.includes(SingleTurnParams.CONTEXT) && tc.context != null)
    data.context = tc.context;
  if (
    params.includes(SingleTurnParams.RETRIEVAL_CONTEXT) &&
    tc.retrievalContext != null
  )
    data.retrieval_context = resolveRetrievalContext(tc.retrievalContext);
  if (params.includes(SingleTurnParams.TOOLS_CALLED) && tc.toolsCalled != null)
    data.tools_called = tc.toolsCalled.map((t: ToolCall) => JSON.stringify(t));
  if (
    params.includes(SingleTurnParams.EXPECTED_TOOLS) &&
    tc.expectedTools != null
  )
    data.expected_tools = tc.expectedTools.map((t: ToolCall) =>
      JSON.stringify(t),
    );
  return JSON.stringify(data, null, 2);
}

/**
 * Mask contestant names with shuffled dummy names + randomize order, and render
 * the arena test case as a JSON blob for the judge. Returns the blob plus the
 * dummy→real name map (used to unmask the winner/reason). Mirrors
 * `format_arena_test_case`.
 */
export function formatArenaTestCase(
  params: SingleTurnParams[],
  testCase: ArenaTestCase,
): [string, Record<string, string>] {
  const realNames = testCase.contestants.map((c) => c.name);
  const fakeNames = shuffled(FAKE_NAMES);
  while (fakeNames.length < realNames.length) {
    fakeNames.push(`Contestant${fakeNames.length + 1}`);
  }

  const dummyToReal: Record<string, string> = {};
  realNames.forEach((real, i) => {
    dummyToReal[fakeNames[i]] = real;
  });
  const realToDummy: Record<string, string> = {};
  for (const [dummy, real] of Object.entries(dummyToReal)) {
    realToDummy[real] = dummy;
  }

  const case0 = testCase.contestants[0].testCase;
  const data: Record<string, unknown> = {};
  if (params.includes(SingleTurnParams.INPUT)) data.input = case0.input;
  if (params.includes(SingleTurnParams.EXPECTED_OUTPUT))
    data.expected_output = case0.expectedOutput;

  const arena: Record<string, string> = {};
  for (const contestant of shuffled(testCase.contestants)) {
    const dummy = realToDummy[contestant.name] ?? contestant.name;
    arena[dummy] = formattedLLMTestCase(params, contestant.testCase);
  }
  data.arena_test_cases = arena;

  return [JSON.stringify(data, null, 2), dummyToReal];
}
