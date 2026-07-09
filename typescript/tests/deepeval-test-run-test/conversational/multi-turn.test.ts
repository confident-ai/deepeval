// Multi-turn evaluation: build a ConversationalTestCase from turns and assert
// it passes real conversational metrics.
import { it } from "vitest";
import { assertTest, ConversationalTestCase, Turn } from "deepeval";
import {
  turnRelevancy,
  professionalism,
  roleAdherence,
} from "../fixtures/metrics";

it("multi-turn conversation passes conversational metrics", async () => {
  const conversation = new ConversationalTestCase({
    chatbotRole: "a polite and helpful customer support agent",
    scenario: "A customer asks for a refund on a recent order.",
    expectedOutcome: "The agent acknowledges the request and initiates a refund.",
    turns: [
      new Turn({
        role: "user",
        content: "Hi, I'd like a refund for my order #123.",
      }),
      new Turn({
        role: "assistant",
        content:
          "Of course, I can help with that. I've started the refund for order #123 — it should arrive within 3–5 business days. Is there anything else I can help you with?",
      }),
      new Turn({ role: "user", content: "No, that's all. Thank you!" }),
      new Turn({
        role: "assistant",
        content: "You're very welcome. Have a great day!",
      }),
    ],
  });

  await assertTest(conversation, [
    turnRelevancy(),
    professionalism(),
    roleAdherence(),
  ]);
});
