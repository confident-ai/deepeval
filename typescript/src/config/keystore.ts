// The `.deepeval/.deepeval` JSON store, ported from deepeval/key_handler.py.
// Non-secret settings live here so they survive across shells without a dotenv
// file.

import * as fs from "fs";
import * as path from "path";
import { HIDDEN_DIR, KEY_FILE } from "@/constants";
import { isSecretSetting } from "@/config/schema";
import { isReadOnlyFileSystem } from "@/config/utils";

export type KeystoreData = Record<string, string>;

function keystorePath(): string {
  return path.join(process.cwd(), HIDDEN_DIR, KEY_FILE);
}

export function readKeystore(): KeystoreData {
  try {
    const raw = fs.readFileSync(keystorePath(), "utf-8");
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as KeystoreData;
    }
  } catch {}
  return {};
}

function writeKeystore(data: KeystoreData): void {
  if (isReadOnlyFileSystem()) return;
  const file = keystorePath();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(data), "utf-8");
}

/** `null` removes a key. Secrets are skipped rather than written in plaintext. */
export function updateKeystore(updates: Record<string, string | null>): void {
  const data = readKeystore();
  let changed = false;

  for (const [key, value] of Object.entries(updates)) {
    if (value === null) {
      if (key in data) {
        delete data[key];
        changed = true;
      }
      continue;
    }
    if (isSecretSetting(key)) {
      console.warn(
        `${key} is a secret setting, refusing to persist it to ` +
          `${HIDDEN_DIR}/${KEY_FILE}. Keep secrets in .env or .env.local instead.`,
      );
      continue;
    }
    if (data[key] !== value) {
      data[key] = value;
      changed = true;
    }
  }

  if (changed) writeKeystore(data);
}

export function keystoreLocation(): string {
  return path.join(HIDDEN_DIR, KEY_FILE);
}
