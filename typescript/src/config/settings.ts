import { z } from "zod";
import { Environment } from "@/tracing/utils";
import { cameFromDotenv, loadDotenvFiles } from "@/config/dotenv-load";
import { DotenvHandler } from "@/config/dotenv-handler";
import {
  keystoreLocation,
  readKeystore,
  updateKeystore,
} from "@/config/keystore";
import {
  getFieldSchema,
  isSecretSetting,
  normalizeSettingKey,
  resolveSettingNames,
  settingsSchema,
  SETTING_NAMES,
  type SettingName,
  type Settings,
} from "@/config/schema";

export type { Settings, SettingName } from "@/config/schema";
export { Environment };

const DEFAULT_DOTENV_PATH = ".env.local";

let _settingsSingleton: Settings | null = null;

/** Precedence: process env (dotenv included, loaded on import), then keystore. */
function rawValues(): Record<string, string> {
  const keystore = readKeystore();
  const raw: Record<string, string> = {};
  for (const name of SETTING_NAMES) {
    const fromEnv = process.env[name];
    if (fromEnv !== undefined && fromEnv !== "") {
      raw[name] = fromEnv;
      continue;
    }
    const fromKeystore = keystore[name];
    if (fromKeystore !== undefined && fromKeystore !== "") {
      raw[name] = fromKeystore;
    }
  }
  return raw;
}

/**
 * Fold the deprecated `DEEPEVAL_TELEMETRY_ENABLED` into the opt-out flag, as
 * Python's `_apply_telemetry_enabled_alias` does: any OFF signal wins.
 */
function applyTelemetryOptOutAlias(settings: Settings): Settings {
  const optOut = settings.DEEPEVAL_TELEMETRY_OPT_OUT;
  const legacyEnabled = settings.DEEPEVAL_TELEMETRY_ENABLED;
  const offSignal = optOut === true || legacyEnabled === false;
  const onSignal = optOut === false || legacyEnabled === true;

  if (offSignal && onSignal && settings.DEEPEVAL_VERBOSE_MODE) {
    console.warn(
      `Conflicting telemetry flags: DEEPEVAL_TELEMETRY_OPT_OUT=${optOut}, ` +
        `DEEPEVAL_TELEMETRY_ENABLED=${legacyEnabled}. Defaulting to off.`,
    );
  }

  return { ...settings, DEEPEVAL_TELEMETRY_OPT_OUT: offSignal };
}

function parseSettings(raw: Record<string, string>): Settings {
  const result = settingsSchema.safeParse(raw);
  if (result.success) return applyTelemetryOptOutAlias(result.data);

  // A bad value in the environment shouldn't be fatal at import time.
  const bad = new Set<string>();
  for (const issue of result.error.issues) {
    const name = issue.path[0];
    if (typeof name === "string") bad.add(name);
  }
  for (const name of bad) {
    const issue = result.error.issues.find((i) => i.path[0] === name);
    console.warn(
      `Ignoring invalid value for ${name}: ${issue?.message ?? "invalid value"}`,
    );
    delete raw[name];
  }
  return applyTelemetryOptOutAlias(settingsSchema.parse(raw));
}

export function getSettings(): Settings {
  if (_settingsSingleton === null) {
    _settingsSingleton = parseSettings(rawValues());
  }
  return _settingsSingleton;
}

export function resetSettings({
  reloadDotenv = false,
}: { reloadDotenv?: boolean } = {}): Settings {
  if (reloadDotenv) loadDotenvFiles();
  _settingsSingleton = null;
  return getSettings();
}

export type SettingSource = "env" | "dotenv" | "keystore" | "default";

export function getSettingSource(name: SettingName): SettingSource {
  const fromEnv = process.env[name];
  if (fromEnv !== undefined && fromEnv !== "") {
    return cameFromDotenv(name) ? "dotenv" : "env";
  }
  const fromKeystore = readKeystore()[name];
  if (fromKeystore !== undefined && fromKeystore !== "") return "keystore";
  return "default";
}

function serialize(value: unknown): string {
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

/** Mask secrets and clip long values for terminal output. */
export function formatSettingValue(name: SettingName, value: unknown): string {
  if (value === undefined || value === null) return "";
  if (isSecretSetting(name)) {
    return String(value).trim() === "" ? "" : "********";
  }
  const text = serialize(value);
  return text.length <= 120 ? text : `${text.slice(0, 117)}…`;
}

export function parseSettingValue(name: SettingName, raw: string): unknown {
  const result = getFieldSchema(name).safeParse(raw);
  if (!result.success) {
    const detail = result.error.issues.map((issue) => issue.message).join("; ");
    throw new SettingsError(`Invalid value for ${name}: '${raw}'. ${detail}`);
  }
  return result.data;
}

export class SettingsError extends Error {}

/** A draft of the settings; assigning `null` unsets a key. */
export type SettingsDraft = {
  -readonly [K in SettingName]: Settings[K] | null;
};

export interface EditOptions {
  /** `dotenv[:path]`, or undefined for "keystore only". */
  save?: string | null;
}

export interface EditResult {
  /** False when a `--save` value was given but isn't a supported target. */
  handled: boolean;
  path: string | null;
  updated: Record<string, string>;
  removed: string[];
}

function parseSaveOption(save: string | null | undefined): {
  handled: boolean;
  path: string | null;
} {
  if (!save) return { handled: false, path: null };
  const [kind, rest] = save.split(/:(.+)/, 2);
  if (kind !== "dotenv") return { handled: false, path: null };
  return {
    handled: true,
    path: rest && rest.trim() ? rest : DEFAULT_DOTENV_PATH,
  };
}

/** An explicit `--save` wins over the `DEEPEVAL_DEFAULT_SAVE` project default. */
export function resolveSaveTarget(
  save: string | null | undefined,
): string | null {
  if (save) return save;
  const fallback = process.env.DEEPEVAL_DEFAULT_SAVE;
  return fallback && fallback.trim() ? fallback.trim() : null;
}

/**
 * The counterpart of Python's `Settings.edit()`. The draft is a Proxy so only
 * the keys `mutate` touches get persisted.
 */
export function editSettings(
  mutate: (draft: SettingsDraft) => void,
  options: EditOptions = {},
): EditResult {
  const touched = new Map<SettingName, unknown>();
  const current = getSettings();

  const draft = new Proxy({} as SettingsDraft, {
    get(_target, prop: string) {
      if (touched.has(prop as SettingName)) {
        return touched.get(prop as SettingName);
      }
      return (current as Record<string, unknown>)[prop];
    },
    set(_target, prop: string, value: unknown) {
      if (!SETTING_NAMES.includes(prop as SettingName)) {
        throw new SettingsError(`Unknown setting: ${prop}`);
      }
      touched.set(prop as SettingName, value);
      return true;
    },
  });

  mutate(draft);

  const updated: Record<string, string> = {};
  const removed: string[] = [];
  for (const [name, value] of touched) {
    if (value === null || value === undefined || value === "") {
      if (rawValues()[name] !== undefined) removed.push(name);
      continue;
    }
    const text = serialize(value);
    if (rawValues()[name] !== text) updated[name] = text;
  }

  // Validate the whole set together, before anything is written.
  const nextRaw = { ...rawValues(), ...updated };
  for (const name of removed) delete nextRaw[name];
  const validated = settingsSchema.safeParse(nextRaw);
  if (!validated.success) {
    const detail = validated.error.issues
      .map((issue) => `${issue.path.join(".")}: ${issue.message}`)
      .join("; ");
    throw new SettingsError(`Invalid settings: ${detail}`);
  }

  for (const [name, text] of Object.entries(updated)) process.env[name] = text;
  for (const name of removed) delete process.env[name];

  const keystoreUpdates: Record<string, string | null> = {};
  for (const [name, text] of Object.entries(updated)) {
    if (!isSecretSetting(name)) keystoreUpdates[name] = text;
  }
  for (const name of removed) keystoreUpdates[name] = null;
  if (Object.keys(keystoreUpdates).length > 0) updateKeystore(keystoreUpdates);

  const target = resolveSaveTarget(options.save);
  const { handled, path } = parseSaveOption(target);
  if (handled && path) {
    const dotenv = new DotenvHandler(path);
    if (Object.keys(updated).length > 0) dotenv.upsert(updated);
    if (removed.length > 0) dotenv.unset(removed);
  }

  _settingsSingleton = applyTelemetryOptOutAlias(validated.data);
  return { handled: handled || !target, path, updated, removed };
}

export {
  SETTING_NAMES,
  isSecretSetting,
  normalizeSettingKey,
  resolveSettingNames,
  keystoreLocation,
};
export type { z };
