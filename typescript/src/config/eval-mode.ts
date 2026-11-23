// Resolves who decides in an LLM-as-a-judge metric: `DEEPEVAL_EVAL_MODE`.
//
// - `llm`: the evaluation LLM runs the whole chain (extraction, decision,
//   reason). The legacy algorithm and the default.
// - `hybrid`: the LLM extracts and writes the reason; a System One model (Jev)
//   answers the decision points that are wired for it. A Jev call that fails
//   at runtime hands that one decision to the LLM. A missing key or SDK still
//   fails.
// - `system_one`: Jev runs the whole metric as one request over the raw test
//   case and the reason is deterministic text built from Jev's answers. No LLM
//   is built or called, and there is no fallback.
//
// Precedence: an explicit `evalMode` on the metric, then the setting, then
// `llm`. Jev is opted into through the eval mode alone; the `DEEPEVAL_MODE`
// feature channel plays no part. Gate code with `resolveEvalMode()` /
// `EvalMode` members, never by comparing raw strings.

import { DEEPEVAL_EVAL_MODE } from "@/constants";
// Cycle (keystore -> schema -> eval-mode) is safe: only used inside a function.
import { readKeystore } from "@/config/keystore";

export const EVAL_MODE_ENV_VAR = DEEPEVAL_EVAL_MODE;

export const EvalMode = {
  LLM: "llm",
  HYBRID: "hybrid",
  SYSTEM_ONE: "system_one",
} as const;

export type EvalModeName = (typeof EvalMode)[keyof typeof EvalMode];

export const SUPPORTED_EVAL_MODES: readonly EvalModeName[] = [
  EvalMode.LLM,
  EvalMode.HYBRID,
  EvalMode.SYSTEM_ONE,
];

export const DEFAULT_EVAL_MODE: EvalModeName = EvalMode.LLM;

/** Whether any Jev call can happen in this mode. */
export function usesSystemOne(mode: EvalModeName): boolean {
  return mode !== EvalMode.LLM;
}

/**
 * The `EvalModeName` for a raw value, or `undefined` if it is unset, blank or
 * unrecognised. Case-insensitive, whitespace ignored, otherwise exact.
 */
export function normalizeEvalMode(
  value: string | undefined | null,
): EvalModeName | undefined {
  if (value === undefined || value === null) return undefined;
  const text = String(value).trim().toLowerCase();
  return (SUPPORTED_EVAL_MODES as readonly string[]).includes(text)
    ? (text as EvalModeName)
    : undefined;
}

/** The configured mode (env first, then the `.deepeval` keystore), if any. */
export function configuredEvalMode(
  env: NodeJS.ProcessEnv = process.env,
): EvalModeName | undefined {
  let raw = env[DEEPEVAL_EVAL_MODE];
  if (raw === undefined || raw.trim() === "") {
    raw = readKeystore()[DEEPEVAL_EVAL_MODE];
  }
  return normalizeEvalMode(raw);
}

/**
 * The effective eval mode for a metric. `override` is the metric's `evalMode`
 * option and wins outright. An unrecognised override throws (the user typed
 * it), whereas an unrecognised setting silently means unset.
 */
export function resolveEvalMode(override?: string | null): EvalModeName {
  if (override !== undefined && override !== null) {
    const mode = normalizeEvalMode(override);
    if (mode === undefined) {
      throw new Error(
        `Unsupported evalMode ${JSON.stringify(override)}. Valid values: ` +
          `${SUPPORTED_EVAL_MODES.join(", ")}.`,
      );
    }
    return mode;
  }
  return configuredEvalMode() ?? DEFAULT_EVAL_MODE;
}
