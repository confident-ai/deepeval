// Comment-preserving dotenv writer, ported from deepeval/cli/dotenv_handler.py.
// The `dotenv` package only reads.

import * as fs from "fs";
import * as path from "path";
import { isReadOnlyFileSystem } from "@/config/utils";

const LINE_RE = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/;

export class DotenvHandler {
  constructor(private readonly filePath: string = ".env.local") {}

  get path(): string {
    return this.filePath;
  }

  private quoteIfNeeded(value: string): string {
    const quoted =
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"));
    if (quoted) return value;
    return /[\s#]/.test(value) ? `"${value}"` : value;
  }

  private readLines(): string[] {
    if (!fs.existsSync(this.filePath)) return [];
    return fs.readFileSync(this.filePath, "utf-8").split("\n");
  }

  private write(lines: string[]): void {
    if (isReadOnlyFileSystem()) return;
    const dir = path.dirname(this.filePath);
    if (dir && dir !== ".") fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      this.filePath,
      lines.join("\n") + (lines.length ? "\n" : ""),
      "utf-8",
    );
    try {
      fs.chmodSync(this.filePath, 0o600);
    } catch {}
  }

  /** Replaces keys in place, preserving comments and ordering. */
  upsert(updates: Record<string, string>): void {
    const lines = this.readLines();
    // Drop the empty element a trailing newline splits into, or the file grows
    // a blank line on every write.
    if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();

    const seen = new Set<string>();
    for (let i = 0; i < lines.length; i++) {
      const match = LINE_RE.exec(lines[i]);
      if (!match) continue;
      const key = match[1];
      if (key in updates && !seen.has(key)) {
        lines[i] = `${key}=${this.quoteIfNeeded(updates[key])}`;
        seen.add(key);
      }
    }

    const toAppend = Object.entries(updates)
      .filter(([key]) => !seen.has(key))
      .map(([key, value]) => `${key}=${this.quoteIfNeeded(value)}`);
    if (toAppend.length > 0) {
      if (lines.length > 0 && lines[lines.length - 1].trim()) lines.push("");
      lines.push(...toAppend);
    }

    this.write(lines);
  }

  unset(keys: Iterable<string>): void {
    if (!fs.existsSync(this.filePath)) return;
    const remove = new Set(keys);
    const lines = this.readLines();
    if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
    const kept = lines.filter((line) => {
      const match = LINE_RE.exec(line);
      return !(match && remove.has(match[1]));
    });
    this.write(kept);
  }
}
