import * as path from "path";
import { pathToFileURL } from "url";
import type { InspectUiModule } from "@/inspect/model";
import {
  InspectLoadError,
  loadTestRun,
  resolveInspectTarget,
  summarizeTestRun,
} from "@/inspect/loader";
import { listTestRuns, parseSource } from "@/sqlite-store/store";

export { InspectLoadError, NoTracesError } from "@/inspect/loader";

export class InspectUnavailableError extends Error {}

// Ink has no CommonJS build, so the UI is compiled as ESM. Building the
// specifier at runtime keeps this program from pulling the `.tsx` sources into
// its own output.
async function loadUi(): Promise<InspectUiModule> {
  const entry = pathToFileURL(path.join(__dirname, "ui", "app.js")).href;
  try {
    return (await import(entry)) as InspectUiModule;
  } catch (e) {
    const message = (e as Error).message;
    if (/Cannot find (module|package)|ERR_MODULE_NOT_FOUND/.test(message)) {
      throw new InspectUnavailableError(
        "`deepeval inspect` needs its terminal UI dependencies:\n\n" +
          "  npm install ink react\n",
      );
    }
    throw e;
  }
}

export interface RunInspectOptions {
  target?: string;
  folder?: string;
  /** Run id inside a SQLite store (`DEEPEVAL_LOCAL_STORE=sqlite`). */
  runId?: number | null;
}

/** `path/deepeval.db#3` keeps the `#3` when made relative for the header. */
function displaySource(source: string): string {
  const parsed = parseSource(source);
  if (!parsed) return path.relative(process.cwd(), source) || source;
  const rel = path.relative(process.cwd(), parsed.dbPath) || parsed.dbPath;
  return parsed.runId === null ? rel : `${rel}#${parsed.runId}`;
}

export async function runInspect(
  options: RunInspectOptions = {},
): Promise<void> {
  const file = resolveInspectTarget(options.target, options.folder, {
    runId: options.runId,
  });
  const traces = loadTestRun(file);
  const summary = summarizeTestRun(file);

  if (!process.stdout.isTTY) {
    throw new InspectLoadError(
      "`deepeval inspect` is an interactive terminal UI and needs a TTY. " +
        "Run it directly in a terminal rather than through a pipe or CI job.",
    );
  }

  const ui = await loadUi();
  await ui.mount({
    traces,
    sourcePath: displaySource(file),
    summary,
  });
}

/** Print the runs in a SQLite store as a table (the `--list` flag). */
export function listStoredRuns(options: RunInspectOptions = {}): void {
  const source = resolveInspectTarget(options.target, options.folder, {
    runId: options.runId,
  });
  const parsed = parseSource(source);
  if (!parsed) {
    throw new InspectLoadError(
      "--list only works with a SQLite store (deepeval.db). " +
        `Resolved source was: ${displaySource(source)}`,
    );
  }
  const runs = listTestRuns(parsed.dbPath, 50);
  if (runs.length === 0) {
    console.log(`${parsed.dbPath} contains no test runs yet.`);
    return;
  }
  const fmt = (v: number | null | undefined, digits: number) =>
    v === null || v === undefined ? "" : v.toFixed(digits);
  console.log(parsed.dbPath);
  console.table(
    runs.map((r) => ({
      id: r.id,
      "created_at (UTC)": (r.created_at ?? "").slice(0, 19).replace("T", " "),
      identifier: r.identifier ?? "",
      passed: r.test_passed ?? "",
      failed: r.test_failed ?? "",
      "duration (s)": fmt(r.run_duration, 2),
      "cost (USD)": fmt(r.evaluation_cost, 4),
    })),
  );
  console.log(
    `Open one with: deepeval inspect ${parsed.dbPath} --run-id <id>`,
  );
}
