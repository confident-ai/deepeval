// Port of `deepeval/telemetry/identity.py`. The store lives in the user's home
// directory, at the path and keys Python uses, so a machine running both SDKs
// reports one id. Reads are synchronous because `capture()` is called from
// non-async code paths.

import * as crypto from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

import { HIDDEN_DIR } from "@/constants";
import { isReadOnlyFileSystem } from "@/config/utils";
// Mutually dependent with `client`, but safe: every use is at call time, never
// during module initialization.
import { telemetryOptOut } from "@/telemetry/client";
import { Feature } from "@/telemetry/events";
import { TelemetryKey, UserStatus } from "@/telemetry/properties";

export const TELEMETRY_DATA_FILE = ".deepeval_telemetry.txt";

export const OPTED_OUT_ID = "telemetry-opted-out";

const NOT_LOGGED_IN = "NA";

export interface Identity {
  anonymousId: string;
  status: UserStatus;
}

let cachedData: Record<string, string> | null = null;
let cachedIdentity: Identity | null = null;

function homeDir(): string {
  const override = process.env.DEEPEVAL_HOME;
  if (override && override.trim() !== "") return override;
  return path.join(os.homedir(), ".deepeval");
}

export function telemetryPath(): string {
  return path.join(homeDir(), TELEMETRY_DATA_FILE);
}

/** Where the id lived before it moved to the home directory. */
function legacyPaths(): string[] {
  return [
    path.join(__dirname, "telemetry.txt"),
    path.join(HIDDEN_DIR, TELEMETRY_DATA_FILE),
    TELEMETRY_DATA_FILE,
  ];
}

function parse(filePath: string): Record<string, string> {
  let contents: string;
  try {
    contents = fs.readFileSync(filePath, "utf-8");
  } catch {
    return {};
  }
  const data: Record<string, string> = {};
  for (const line of contents.split("\n")) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    const separator = trimmed.indexOf("=");
    const key = separator === -1 ? trimmed : trimmed.slice(0, separator);
    const value = separator === -1 ? "" : trimmed.slice(separator + 1);
    if (key) data[key] = value;
  }
  return data;
}

function legacyFeatureKey(feature: Feature): string {
  return `DEEPEVAL_${feature.toUpperCase()}_STATUS`;
}

/**
 * Fold the pre-v2 per-feature keys into the single seen-features list, so
 * existing users do not all report `feature.status = new` once more.
 */
function adoptLegacyFeatureKeys(data: Record<string, string>): boolean {
  const seen = seenFeatures(data);
  let changed = false;
  for (const feature of Object.values(Feature)) {
    // Presence is the signal, whatever the recorded value.
    const key = legacyFeatureKey(feature);
    if (!(key in data)) continue;
    delete data[key];
    changed = true;
    seen.add(feature);
  }
  if (changed) {
    data[TelemetryKey.SEEN_FEATURES] = [...seen].sort().join(",");
  }
  return changed;
}

function load(): Record<string, string> {
  const current = telemetryPath();
  if (fs.existsSync(current)) {
    const data = parse(current);
    if (adoptLegacyFeatureKeys(data)) persist(data);
    return data;
  }
  for (const legacy of legacyPaths()) {
    if (!fs.existsSync(legacy)) continue;
    const data = parse(legacy);
    if (data[TelemetryKey.ID]) {
      adoptLegacyFeatureKeys(data);
      persist(data);
      return data;
    }
  }
  return {};
}

function persist(data: Record<string, string>): void {
  if (telemetryOptOut() || isReadOnlyFileSystem()) return;
  try {
    const filePath = telemetryPath();
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    const lines = Object.entries(data).map(
      ([key, value]) => `${key}=${value}\n`,
    );
    fs.writeFileSync(filePath, lines.join(""), "utf-8");
  } catch {
    // A read-only or unwritable home must never break an evaluation.
  }
}

export function readTelemetryFile(): Record<string, string> {
  if (cachedData === null) cachedData = load();
  return { ...cachedData };
}

export function writeTelemetryFile(data: Record<string, string>): void {
  cachedData = { ...data };
  persist(cachedData);
}

/** Resolves the id and first-run status together, so only one event is new. */
export function getIdentity(): Identity {
  if (telemetryOptOut()) {
    return { anonymousId: OPTED_OUT_ID, status: UserStatus.OLD };
  }
  if (cachedIdentity !== null) return cachedIdentity;

  const data = readTelemetryFile();
  let anonymousId = data[TelemetryKey.ID];
  let status: UserStatus;
  if (anonymousId) {
    status = UserStatus.OLD;
  } else {
    anonymousId = crypto.randomUUID();
    status = UserStatus.NEW;
    data[TelemetryKey.ID] = anonymousId;
    // Persist "old" now so the next process reports old, not this one.
    data[TelemetryKey.STATUS] = UserStatus.OLD;
    writeTelemetryFile(data);
  }

  cachedIdentity = { anonymousId, status };
  return cachedIdentity;
}

export function getUniqueId(): string {
  return getIdentity().anonymousId;
}

export function getStatus(): string {
  return getIdentity().status;
}

function seenFeatures(data: Record<string, string>): Set<string> {
  const raw = data[TelemetryKey.SEEN_FEATURES] ?? "";
  return new Set(raw.split(",").filter((item) => item !== ""));
}

export function getFeatureStatus(feature: Feature): UserStatus {
  const data = readTelemetryFile();
  return seenFeatures(data).has(feature) ? UserStatus.OLD : UserStatus.NEW;
}

export function setLastFeature(feature: Feature): void {
  const data = readTelemetryFile();
  data[TelemetryKey.LAST_FEATURE] = feature;
  const seen = seenFeatures(data);
  seen.add(feature);
  data[TelemetryKey.SEEN_FEATURES] = [...seen].sort().join(",");
  writeTelemetryFile(data);
}

export function getLastFeature(): Feature {
  const data = readTelemetryFile();
  const last = data[TelemetryKey.LAST_FEATURE];
  const known = Object.values(Feature).find((value) => value === last);
  return known ?? Feature.UNKNOWN;
}

/** Local only: the address is never transmitted, just the `isLoggedIn` flag. */
export function setLoggedInWith(loggedInWith: string): void {
  const data = readTelemetryFile();
  data[TelemetryKey.LOGGED_IN_WITH] = loggedInWith;
  writeTelemetryFile(data);
}

export function getLoggedInWith(): string {
  const data = readTelemetryFile();
  return data[TelemetryKey.LOGGED_IN_WITH] ?? NOT_LOGGED_IN;
}

export function isLoggedIn(): boolean {
  return getLoggedInWith() !== NOT_LOGGED_IN;
}

export function resetCacheForTesting(): void {
  cachedData = null;
  cachedIdentity = null;
}
