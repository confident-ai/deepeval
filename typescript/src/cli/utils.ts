import { createInterface } from "readline";
import { Writable } from "stream";
import { CommanderError } from "commander";
import type { EditResult } from "@/config/settings";

export function badParameter(message: string): never {
  throw new CommanderError(1, "deepeval.badParameter", message);
}

export interface SaveResultOptions {
  result: EditResult;
  save?: string | null;
  quiet?: boolean;
  successMessage?: string;
  updatedMessage?: string;
  noChangesMessage?: string;
}

/** Returns whether anything was printed, so callers can add follow-up lines. */
export function handleSaveResult({
  result,
  save,
  quiet,
  successMessage,
  updatedMessage = "Saved environment variables to {path} (ensure it's git-ignored).",
  noChangesMessage = "No changes to save in {path}.",
}: SaveResultOptions): boolean {
  if (!result.handled && save) {
    badParameter("Unsupported --save option. Use --save=dotenv[:path].");
  }
  if (quiet) return false;

  const changed =
    Object.keys(result.updated).length > 0 || result.removed.length > 0;
  if (result.path && changed) {
    console.log(updatedMessage.replace("{path}", result.path));
  } else if (result.path) {
    console.log(noChangesMessage.replace("{path}", result.path));
  }

  if (successMessage) console.log(successMessage);
  return true;
}

export const SAVE_OPTION_HELP =
  "Persist CLI parameters as environment variables in a dotenv file. " +
  "Usage: --save=dotenv[:path] (default: .env.local)";

export const QUIET_OPTION_HELP =
  "Suppress printing to the terminal (useful for CI).";

export function coerceBlankToNull(
  value: string | undefined | null,
): string | null {
  if (value === undefined || value === null) return null;
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

/** Prompt on stdin. `hidden` suppresses echo for secrets. */
export function prompt(question: string, hidden = false): Promise<string> {
  let muted = false;
  const output = new Writable({
    write(chunk, encoding, callback) {
      if (!muted) process.stdout.write(chunk, encoding);
      callback();
    },
  });

  const rl = createInterface({
    input: process.stdin,
    output,
    terminal: true,
  });

  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      if (hidden) process.stdout.write("\n");
      rl.close();
      resolve(answer.trim());
    });
    muted = hidden;
  });
}

/** Render a left-aligned table with a bold purple header, wrapping long cells. */
export function printTable(
  headers: string[],
  rows: string[][],
  title?: string,
): void {
  const PURPLE = "\x1b[38;2;106;0;255m";
  const BOLD = "\x1b[1m";
  const RESET = "\x1b[0m";
  const terminal = process.stdout.columns || 100;

  const widths = headers.map((header, i) =>
    Math.max(header.length, ...rows.map((row) => (row[i] ?? "").length)),
  );
  const fixed = widths.slice(0, -1).reduce((a, b) => a + b + 3, 0);
  widths[widths.length - 1] = Math.max(
    12,
    Math.min(widths[widths.length - 1], terminal - fixed - 3),
  );

  const line = (cells: string[]): string =>
    cells
      .map((cell, i) => cell.padEnd(widths[i]))
      .join("   ")
      .trimEnd();

  if (title) console.log(`\n${BOLD}${title}${RESET}`);
  // Colour the whole line, so padding is computed on plain text.
  console.log(`${PURPLE}${BOLD}${line(headers)}${RESET}`);
  console.log(widths.map((width) => "─".repeat(width)).join("───"));

  for (const row of rows) {
    const wrapped = row.map((cell, i) => wrapCell(cell, widths[i]));
    const height = Math.max(...wrapped.map((cell) => cell.length));
    for (let l = 0; l < height; l++) {
      console.log(line(wrapped.map((cell) => cell[l] ?? "")));
    }
  }
  console.log();
}

function wrapCell(text: string, width: number): string[] {
  if (text.length <= width) return [text];
  const out: string[] = [];
  let current = "";
  for (const word of text.split(/\s+/)) {
    if (current === "") {
      current = word;
    } else if (current.length + 1 + word.length <= width) {
      current += ` ${word}`;
    } else {
      out.push(current);
      current = word;
    }
    while (current.length > width) {
      out.push(current.slice(0, width));
      current = current.slice(width);
    }
  }
  out.push(current);
  return out;
}
