import { z } from "zod";

const nonEmpty = z.string().trim().min(1);

/** One message: what the user sent or what the bot replied. `==text==` highlights part of it. */
const message = z.strictObject({
  from: z.enum(["user", "bot"]),
  text: nonEmpty,
  /** Marks a reply as one the bot should (pass) or shouldn't (fail) have given. */
  verdict: z.enum(["pass", "fail"]).optional(),
});

/**
 * A mock chat with an unbranded bot: never a real product's name, logo or look,
 * since these often show a bot failing (a jailbreak, a leak) that no real product did.
 */
export const chatData = z.strictObject({
  /** The bot's name, e.g. "Chatbot" or "Command Bot". */
  bot: nonEmpty.default("Chatbot"),
  messages: z.array(message).min(2),
});

export type ChatData = z.infer<typeof chatData>;
export type ChatMessage = ChatData["messages"][number];

/** `chat`: bubbles in a chat window. `playground`: a prompt field, a Generate button and an output panel. */
export const CHAT_TYPES = ["chat", "playground"] as const;
export type ChatType = (typeof CHAT_TYPES)[number];

export const chatProblems = (data: ChatData): string[] => {
  const problems: string[] = [];
  if (!data.messages.some(m => m.from === "user") || !data.messages.some(m => m.from === "bot")) {
    problems.push("needs at least one user message and one bot reply");
  }
  data.messages.forEach((m, i) => {
    if (m.verdict && m.from === "user") problems.push(`"messages.${i}": only bot replies take a "verdict"`);
  });
  return problems;
};

export const chatTypeProblems = (type: ChatType, data: ChatData): string[] =>
  type === "playground" && data.messages[0].from !== "user" ? ["a playground starts with the user's prompt"] : [];
