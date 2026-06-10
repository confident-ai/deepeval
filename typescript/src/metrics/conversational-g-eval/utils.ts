import { ConversationalTestCase, MultiTurnParams } from "../../test-case";

/** Display labels for conversational eval params (Python CONVERSATIONAL_G_EVAL_PARAMS). */
export const CONVERSATIONAL_G_EVAL_PARAMS: Partial<
  Record<MultiTurnParams, string>
> = {
  [MultiTurnParams.CONTENT]: "Content",
  [MultiTurnParams.ROLE]: "Role",
  [MultiTurnParams.METADATA]: "Metadata",
  [MultiTurnParams.TAGS]: "Tags",
  [MultiTurnParams.TOOLS_CALLED]: "Tools Called",
  [MultiTurnParams.RETRIEVAL_CONTEXT]: "Retrieval Context",
  [MultiTurnParams.EXPECTED_OUTCOME]: "Expected Outcome",
  [MultiTurnParams.SCENARIO]: "Scenario",
};

/** Join the chosen turn params: "A", "A and B", or "A, B, and C". */
export function constructConversationalGEvalTurnParamsString(
  params: MultiTurnParams[],
): string {
  const labels = params.map((p) => CONVERSATIONAL_G_EVAL_PARAMS[p] ?? p);
  if (labels.length === 1) return labels[0];
  if (labels.length === 2) return labels.join(" and ");
  return `${labels.slice(0, -1).join(", ")}, and ${labels[labels.length - 1]}`;
}

// Turn-level params are rendered separately (as the `turns` list); these are the
// conversation-level fields shown as a block.
const TURN_LEVEL = new Set<MultiTurnParams>([
  MultiTurnParams.RETRIEVAL_CONTEXT,
  MultiTurnParams.TOOLS_CALLED,
  MultiTurnParams.CONTENT,
  MultiTurnParams.ROLE,
]);

function conversationLevelValue(
  testCase: ConversationalTestCase,
  param: MultiTurnParams,
): unknown {
  switch (param) {
    case MultiTurnParams.SCENARIO:
      return testCase.scenario;
    case MultiTurnParams.EXPECTED_OUTCOME:
      return testCase.expectedOutcome;
    case MultiTurnParams.METADATA:
      return testCase.additionalMetadata;
    case MultiTurnParams.TAGS:
      return testCase.tags;
    case MultiTurnParams.CONTEXT:
      return testCase.context;
    case MultiTurnParams.USER_DESCRIPTION:
      return testCase.userDescription;
    case MultiTurnParams.CHATBOT_ROLE:
      return testCase.chatbotRole;
    default:
      return undefined;
  }
}

/** Render the conversation-level (non-turn) fields as a text block, or "". */
export function constructNonTurnsTestCaseString(
  params: MultiTurnParams[],
  testCase: ConversationalTestCase,
): string {
  let body = "";
  for (const param of params) {
    if (TURN_LEVEL.has(param)) continue;
    const value = conversationLevelValue(testCase, param);
    body += `${CONVERSATIONAL_G_EVAL_PARAMS[param] ?? param}:\n${value ?? "None"} \n\n`;
  }
  if (!body) return "";
  return `Conversation-level fields:\n${body}`;
}
