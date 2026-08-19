import * as path from "path";
import { pathToFileURL } from "url";
import type { InspectUiModule } from "@/inspect/model";
import {
  InspectLoadError,
  loadTestRun,
  resolveInspectTarget,
  summarizeTestRun,
} from "@/inspect/loader";

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
}

export async function runInspect(
  options: RunInspectOptions = {},
): Promise<void> {
  const file = resolveInspectTarget(options.target, options.folder);
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
    sourcePath: path.relative(process.cwd(), file) || file,
    summary,
  });
}
