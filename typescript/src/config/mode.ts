// Resolves `DEEPEVAL_MODE`: `stable` (default) or `experimental`.
//
// `experimental` opts a user into the newest deepeval features before they
// are finalised; anything gated on it may change or break between releases.
// `stable` is the default and the fallback for an unset or unrecognised value,
// so a typo can never silently enrol someone.
//
// Gate features with `isExperimental()` rather than comparing strings.

import { DEEPEVAL_MODE } from "@/constants";
// Cycle (keystore -> schema -> mode) is safe: only used inside a function.
import { readKeystore } from "@/config/keystore";

export const MODE_STABLE = "stable";
export const MODE_EXPERIMENTAL = "experimental";
export type DeepEvalMode = typeof MODE_STABLE | typeof MODE_EXPERIMENTAL;

export const DEFAULT_MODE: DeepEvalMode = MODE_STABLE;
export const SUPPORTED_MODES: readonly DeepEvalMode[] = [
  MODE_STABLE,
  MODE_EXPERIMENTAL,
];

const warnedUnrecognised = new Set<string>();

/**
 * Normalize a raw `DEEPEVAL_MODE` value. Returns `undefined` for values that
 * are unset, blank or unrecognised.
 */
export function normalizeDeepEvalMode(
  value: string | undefined | null,
): DeepEvalMode | undefined {
  if (value === undefined || value === null) return undefined;
  const s = value.trim().toLowerCase();
  if (s === MODE_STABLE) return MODE_STABLE;
  if (s === MODE_EXPERIMENTAL) return MODE_EXPERIMENTAL;
  return undefined;
}

/**
 * `"stable"` (default) or `"experimental"`.
 *
 * Mirrors Python's `resolve_deepeval_mode()`: the raw value comes from
 * `process.env` first, then the `.deepeval` keystore (the CLI's `set-mode`
 * writes there when `--save` is omitted). An unrecognised value warns once and
 * falls back to `stable`.
 */
export function resolveDeepEvalMode(
  env: NodeJS.ProcessEnv = process.env,
): DeepEvalMode {
  let raw = env[DEEPEVAL_MODE];
  if (raw === undefined || raw.trim() === "") {
    raw = readKeystore()[DEEPEVAL_MODE];
  }
  const mode = normalizeDeepEvalMode(raw);
  if (mode !== undefined) return mode;
  if (raw !== undefined && raw.trim() !== "" && !warnedUnrecognised.has(raw)) {
    warnedUnrecognised.add(raw);
    console.warn(
      `Warning: unrecognised ${DEEPEVAL_MODE}=${JSON.stringify(raw)}; ` +
        `falling back to '${DEFAULT_MODE}'. Valid values: ${SUPPORTED_MODES.join(", ")}.`,
    );
  }
  return DEFAULT_MODE;
}

/** True when the user has opted into experimental deepeval features. */
export function isExperimental(): boolean {
  return resolveDeepEvalMode() === MODE_EXPERIMENTAL;
}
