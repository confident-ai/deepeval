import { z } from "zod";

const nonEmpty = z.string().trim().min(1);
const index = z.number().int().min(0);

export const TRANSCRIPT_ROLES = ["user", "assistant", "system", "tool"] as const;

const turn = z.strictObject({
  role: z.enum(TRANSCRIPT_ROLES),
  text: nonEmpty,
  /** Overrides the role's label, e.g. "Command Bot". */
  name: nonEmpty.optional(),
  verdict: z.enum(["pass", "fail"]).optional(),
});

export const transcriptData = z.strictObject({
  /** Label for every assistant turn that has no name of its own. */
  speaker: nonEmpty.optional(),
  turns: z.array(turn).default([]),
  panels: z.array(z.strictObject({ title: nonEmpty, turns: z.array(turn).min(1) })).default([]),
  ranges: z.array(z.strictObject({ from: index, to: index, label: nonEmpty })).default([]),
  pointers: z.array(z.strictObject({ turn: index, label: nonEmpty })).default([]),
  judge: z
    .strictObject({
      label: nonEmpty.default("LLM-as-a-Judge"),
      prompt: nonEmpty,
      verdict: nonEmpty,
      verdictLabel: nonEmpty.default("Verdict"),
    })
    .optional(),
});

export type TranscriptData = z.infer<typeof transcriptData>;
export type TranscriptTurn = TranscriptData["turns"][number];

export const TRANSCRIPT_TYPES = ["chat", "turns", "judge", "compare"] as const;
export type TranscriptType = (typeof TRANSCRIPT_TYPES)[number];

export const roleLabel = (turn: TranscriptTurn, speaker?: string): string =>
  turn.name ??
  (turn.role === "assistant" ? (speaker ?? "Assistant") : turn.role === "user" ? "User" : turn.role === "tool" ? "Tool" : "System");

export const transcriptProblems = (data: TranscriptData): string[] => {
  const problems: string[] = [];
  if (!data.turns.length && !data.panels.length) problems.push('needs "turns" or "panels"');
  const last = data.turns.length - 1;
  data.ranges.forEach((r, i) => {
    if (r.from > r.to) problems.push(`"ranges.${i}" starts after it ends`);
    if (r.to > last) problems.push(`"ranges.${i}.to" is turn ${r.to}, but the last turn is ${last}`);
  });
  data.pointers.forEach((p, i) => {
    if (p.turn > last) problems.push(`"pointers.${i}.turn" is turn ${p.turn}, but the last turn is ${last}`);
  });
  return problems;
};

export const transcriptTypeProblems = (type: TranscriptType, data: TranscriptData): string[] => {
  if (type === "chat") {
    return data.turns.some(t => t.role === "user") && data.turns.some(t => t.role === "assistant")
      ? []
      : ["chat needs at least one user turn and one assistant turn"];
  }
  if (type === "turns") return data.turns.length >= 2 ? [] : [`turns needs at least 2 turns, this data has ${data.turns.length}`];
  if (type === "judge") return data.judge ? [] : ['judge needs a "judge" with a prompt and a verdict'];
  return data.panels.length >= 2 ? [] : [`compare needs at least 2 panels, this data has ${data.panels.length}`];
};
