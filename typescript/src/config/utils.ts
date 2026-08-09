// Value coercion shared by every env var reader, mirroring Python's
// `deepeval/config/utils.py`. Dependency-free to avoid import cycles.

const TRUTHY = new Set([
  "1",
  "true",
  "t",
  "yes",
  "y",
  "on",
  "enable",
  "enabled",
]);
const FALSY = new Set([
  "0",
  "false",
  "f",
  "no",
  "n",
  "off",
  "disable",
  "disabled",
]);

/** Strip whitespace, then one layer of quotes, then lowercase. */
function normalizeToken(value: string): string {
  const trimmed = value.trim();
  const unquoted =
    trimmed.length >= 2 &&
    (trimmed.startsWith('"') || trimmed.startsWith("'")) &&
    trimmed[0] === trimmed[trimmed.length - 1]
      ? trimmed.slice(1, -1)
      : trimmed;
  return unquoted.trim().toLowerCase();
}

/**
 * `undefined` for an unset, empty, or unrecognized value, so callers can pick
 * between warning and silently falling back.
 */
export function parseBool(value: string | undefined): boolean | undefined {
  if (value === undefined) return undefined;
  const token = normalizeToken(value);
  if (token === "") return undefined;
  if (TRUTHY.has(token)) return true;
  if (FALSY.has(token)) return false;
  return undefined;
}

/** `undefined` when unset, empty, or unparseable. Range is the caller's. */
export function parseNumber(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const trimmed = value.trim();
  if (trimmed === "") return undefined;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/** The canonical string form for a boolean env var, matching Python. */
export function boolToEnvStr(value: boolean): string {
  return value ? "1" : "0";
}

const READ_ONLY_ALIASES = new Set(["READ_ONLY", "READ-ONLY", "READONLY", "RO"]);

/** `READ_ONLY` and its friendly aliases, or `undefined` if not recognized. */
export function normalizeFileSystemMode(
  value: string | undefined,
): "READ_ONLY" | undefined {
  if (value === undefined) return undefined;
  return READ_ONLY_ALIASES.has(value.trim().toUpperCase())
    ? "READ_ONLY"
    : undefined;
}

/**
 * Suppresses DeepEval's own bookkeeping: the keystore, dotenv persistence, the
 * metric cache, the latest test run, the results export. The per-run temp
 * directory is exempt — `deepeval test run` passes worker results through it.
 */
export function isReadOnlyFileSystem(): boolean {
  return (
    normalizeFileSystemMode(process.env.DEEPEVAL_FILE_SYSTEM) === "READ_ONLY"
  );
}

/** Human-readable token list, for error messages. */
export const BOOL_TOKENS_MESSAGE =
  "Expected a boolean: 1/0, true/false, yes/no, on/off, enable(d)/disable(d).";
