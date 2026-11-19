// Context-budget pre-check for System One requests.
//
// Jev has a bounded context window and the SDK ships neither a tokenizer nor a
// dedicated exception for overflow (it surfaces as a 400/422 the retry policy
// rightly refuses to retry). Estimating the size before the call gives the
// user a clear "too large for Jev" error instead of an opaque 400/422.

import { DeepEvalError } from "@/errors";
import {
  JEV_MAX_REQUEST_TOKENS,
  JEV_MAX_STATE_TOKENS,
} from "@/models/system-one/constants";

// Characters per token. English prose averages roughly 4; JSON with its
// punctuation and short keys tokenizes denser, so 3 keeps the estimate on the
// high side and the pre-check conservative.
const CHARS_PER_TOKEN = 3;

export class SystemOneContextLimitError extends DeepEvalError {
  readonly estimatedTokens: number;
  readonly limitTokens: number;

  constructor(
    message: string,
    options: { estimatedTokens: number; limitTokens: number },
  ) {
    super(message);
    this.name = "SystemOneContextLimitError";
    this.estimatedTokens = options.estimatedTokens;
    this.limitTokens = options.limitTokens;
  }
}

// Serialized with Python's default `json.dumps` separators so the estimate,
// and therefore where the budget trips, matches the Python SDK.
function pythonJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "NaN";
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(", ")}]`;
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `${JSON.stringify(k)}: ${pythonJson(v)}`);
    return `{${entries.join(", ")}}`;
  }
  return JSON.stringify(String(value));
}

/** Upper-bound token estimate for a JSON-serialisable value. */
export function estimateTokens(value: unknown): number {
  if (value === null || value === undefined) return 0;
  const text = typeof value === "string" ? value : pythonJson(value);
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/**
 * Throw `SystemOneContextLimitError` if the request is likely to exceed the
 * model's context budget.
 */
export function checkContextBudget(
  state: unknown,
  questions: Record<string, unknown>,
  {
    maxStateTokens = JEV_MAX_STATE_TOKENS,
    maxRequestTokens = JEV_MAX_REQUEST_TOKENS,
  }: { maxStateTokens?: number; maxRequestTokens?: number } = {},
): void {
  const stateTokens = estimateTokens(state);
  const questionTokens = Object.values(questions).map(estimateTokens);
  const longest = questionTokens.length > 0 ? Math.max(...questionTokens) : 0;
  if (stateTokens + longest > maxStateTokens) {
    throw new SystemOneContextLimitError(
      `System One state (${stateTokens} est. tokens) plus the longest ` +
        `question (${longest}) exceeds the ${maxStateTokens}-token state ` +
        `budget. Send less state or judge this test case with the LLM.`,
      { estimatedTokens: stateTokens + longest, limitTokens: maxStateTokens },
    );
  }
  const total = stateTokens + questionTokens.reduce((a, b) => a + b, 0);
  if (total > maxRequestTokens) {
    throw new SystemOneContextLimitError(
      `System One request (${total} est. tokens across ` +
        `${questionTokens.length} questions) exceeds the ` +
        `${maxRequestTokens}-token request budget. Ask fewer questions ` +
        `per request or judge this test case with the LLM.`,
      { estimatedTokens: total, limitTokens: maxRequestTokens },
    );
  }
}
