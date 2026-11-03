import {
  chmod,
  lstat,
  readFile,
  rename,
  unlink,
  writeFile,
} from "node:fs/promises";
import { basename, join } from "node:path";

import type { CommandRunner } from "./git.js";

export const CONFIDENT_API_KEY = "CONFIDENT_API_KEY";

const assignmentPattern = (key: string): RegExp =>
  new RegExp(`^\\s*(?:export\\s+)?${key}\\s*=`);

/** Replace each key in place when present, otherwise append it. */
export const mergeEnvContent = (
  content: string,
  values: Record<string, string>,
): string => {
  const newline = content.includes("\r\n") ? "\r\n" : "\n";
  let merged = content ? content.split(/\r?\n/) : [];
  let appended = false;

  for (const [key, value] of Object.entries(values)) {
    const matches = assignmentPattern(key);
    const assignment = `${key}=${JSON.stringify(value)}`;
    let kept = false;
    merged = merged.filter((line) => {
      if (!matches.test(line)) return true;
      if (kept) return false;
      kept = true;
      return true;
    });
    if (kept) {
      merged[merged.findIndex((line) => matches.test(line))] = assignment;
      continue;
    }
    // One blank line separates newly appended keys from existing content.
    if (!appended && merged.length > 0 && merged.at(-1) !== "") merged.push("");
    merged.push(assignment);
    appended = true;
  }

  while (merged.at(-1) === "") merged.pop();
  return `${merged.join(newline)}${newline}`;
};

/**
 * Parse the subset of dotenv syntax this wizard writes and DeepEval reads, so
 * an existing key can be detected without asking the user to paste it again.
 */
export const parseEnvContent = (content: string): Record<string, string> => {
  const values: Record<string, string> = {};
  for (const line of content.split(/\r?\n/)) {
    const match = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$/.exec(
      line,
    );
    if (!match) continue;
    const raw = match[2]!.trim();
    const quoted = /^(["'])([\s\S]*)\1$/.exec(raw);
    values[match[1]!] = quoted
      ? quoted[1] === '"'
        ? quoted[2]!.replace(/\\(["\\nrt])/g, (_, escaped: string) =>
            escaped === "n"
              ? "\n"
              : escaped === "r"
                ? "\r"
                : escaped === "t"
                  ? "\t"
                  : escaped,
          )
        : quoted[2]!
      : raw.replace(/\s+#.*$/, "").trim();
  }
  return values;
};

const readIfPresent = async (path: string): Promise<string> => {
  try {
    return await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return "";
    throw error;
  }
};

/** Values already in `.env.local`, so detection sees what DeepEval will load. */
export const readEnvValues = async (
  cwd: string,
): Promise<Record<string, string>> =>
  parseEnvContent(await readIfPresent(join(cwd, ".env.local")));

export const writeEnvValues = async (
  cwd: string,
  values: Record<string, string>,
): Promise<string> => {
  const path = join(cwd, ".env.local");
  try {
    const stats = await lstat(path);
    if (stats.isSymbolicLink()) {
      throw new Error("Refusing to replace a symbolic-link .env.local file.");
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }

  const content = mergeEnvContent(await readIfPresent(path), values);
  const temporaryPath = join(
    cwd,
    `.${basename(path)}.${process.pid}.${Date.now()}.tmp`,
  );
  try {
    await writeFile(temporaryPath, content, { mode: 0o600, flag: "wx" });
    await rename(temporaryPath, path);
    await chmod(path, 0o600);
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined);
    throw error;
  }
  return path;
};

const defaultIgnoreCheck = async (
  cwd: string,
  runner: CommandRunner,
): Promise<boolean> => {
  try {
    await runner("git", ["check-ignore", "-q", ".env.local"], { cwd });
    return true;
  } catch {
    return false;
  }
};

export const ensureEnvLocalIgnored = async (
  cwd: string,
  runner: CommandRunner,
  ignoreCheck: (
    cwd: string,
    runner: CommandRunner,
  ) => Promise<boolean> = defaultIgnoreCheck,
): Promise<boolean> => {
  if (await ignoreCheck(cwd, runner)) return false;

  const path = join(cwd, ".gitignore");
  const content = await readIfPresent(path);
  const newline = content.includes("\r\n") ? "\r\n" : "\n";
  const next = `${content}${content && !content.endsWith("\n") ? newline : ""}.env.local${newline}`;
  await writeFile(path, next, { encoding: "utf8", mode: 0o644 });
  return true;
};
