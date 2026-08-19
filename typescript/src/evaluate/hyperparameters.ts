// Port of deepeval/test_run/hyperparameters.py. Hyperparameters label a test run
// with the model, prompt and settings that produced it, so runs can be compared
// on Confident AI.

import * as fs from "fs";
import * as path from "path";
import { DEEPEVAL_RESULTS_DIR, HYPERPARAMETERS_FILE } from "@/constants";
import { Prompt } from "@/prompt";

/** A `Prompt` is sent by reference (`{id, type}`); everything else as a string. */
export type HyperparameterValue = string | number | boolean | Prompt;

export type ProcessedHyperparameter = string | { id: string; type: string };

export type Hyperparameters = Record<
  string,
  HyperparameterValue | null | undefined
>;

export type ProcessedHyperparameters = Record<string, ProcessedHyperparameter>;

/**
 * Normalize user hyperparameters into the shape the platform accepts. Returns
 * `undefined` for absent input, which is what triggers the "no hyperparameters
 * logged" warning at the end of a run.
 */
export function processHyperparameters(
  hyperparameters?: Hyperparameters,
): ProcessedHyperparameters | undefined {
  if (hyperparameters == null) return undefined;

  const processed: ProcessedHyperparameters = {};
  for (const [key, value] of Object.entries(hyperparameters)) {
    if (value == null) continue;

    if (value instanceof Prompt) {
      // An unpulled prompt has no hash or type to reference on the platform.
      // Python pushes it here; we cannot, because this is a sync path.
      if (value.type == null) {
        console.warn(
          `Hyperparameter '${key}' is a Prompt that has not been pulled or ` +
            `pushed, so it has no version to reference — skipping. Call ` +
            `await prompt.pull() before passing it as a hyperparameter.`,
        );
        continue;
      }
      processed[key] = { id: value.hash, type: value.type };
      continue;
    }

    if (
      typeof value !== "string" &&
      typeof value !== "number" &&
      typeof value !== "boolean"
    ) {
      throw new TypeError(
        `Hyperparameter '${key}' must be a string, number, boolean or Prompt.`,
      );
    }
    processed[key] = String(value);
  }

  return processed;
}

/** How many of the logged hyperparameters are prompts (drives the warning tier). */
export function countPrompts(
  hyperparameters?: ProcessedHyperparameters,
): number {
  if (!hyperparameters) return 0;
  return Object.values(hyperparameters).filter((v) => typeof v === "object")
    .length;
}

// --- `deepeval test run` path ----------------------------------------------
// Test files run in Vitest workers, so a `logHyperparameters()` call cannot
// reach the wrap-up in the main process by memory. Each worker writes the same
// JSON file in the shared results directory instead, mirroring how Python's
// `log_hyperparameters` decorator writes to its temp test run file.

function hyperparametersPath(resultsDir: string): string {
  return path.join(resultsDir, HYPERPARAMETERS_FILE);
}

export function writeHyperparameters(
  resultsDir: string,
  hyperparameters: ProcessedHyperparameters,
): void {
  fs.mkdirSync(resultsDir, { recursive: true });
  fs.writeFileSync(
    hyperparametersPath(resultsDir),
    JSON.stringify(hyperparameters),
    "utf-8",
  );
}

export function readHyperparameters(
  resultsDir: string,
): ProcessedHyperparameters | undefined {
  try {
    const raw = fs.readFileSync(hyperparametersPath(resultsDir), "utf-8");
    return JSON.parse(raw) as ProcessedHyperparameters;
  } catch {
    return undefined;
  }
}

/**
 * Attach hyperparameters to the current `deepeval test run`. Counterpart to
 * Python's `@deepeval.log_hyperparameters` decorator; call it once anywhere in a
 * test file. Outside a `deepeval test run` there is no test run to label, so the
 * call is a no-op — pass `hyperparameters` to `evaluate()` instead.
 *
 * @example
 * logHyperparameters({ model: "gpt-4o-mini", temperature: 0.7, prompt });
 */
export function logHyperparameters(hyperparameters: Hyperparameters): void {
  const resultsDir = process.env[DEEPEVAL_RESULTS_DIR];
  if (!resultsDir) return;
  const processed = processHyperparameters(hyperparameters);
  if (!processed) return;
  // Last writer wins, as in Python — one set of hyperparameters per test run.
  writeHyperparameters(resultsDir, processed);
}
