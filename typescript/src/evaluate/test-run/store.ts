import * as fs from "fs";
import * as path from "path";
import { DEEPEVAL_RESULTS_DIR } from "@/constants";
import { getDisplay, type TestRunDisplay } from "@/env-flags";
import {
  buildTestCaseEntry,
  postPersistedTestRun,
  type PersistedCase,
} from "@/evaluate/confident";
import {
  printHyperparametersWarning,
  printResultsTable,
} from "@/evaluate/console-report";
import { readHyperparameters } from "@/evaluate/hyperparameters";
import {
  exportTestRunJson,
  saveLatestTestRun,
  summarizeCases,
  type LocalTestRun,
} from "@/evaluate/test-run/local";
import { EvaluatedCase, MetricData, TestResult } from "@/evaluate/types";

export function getResultsDir(): string | null {
  return process.env[DEEPEVAL_RESULTS_DIR] || null;
}

let workerCaseCount = 0;
function workerFilePath(dir: string): string {
  return path.join(dir, `worker-${process.pid}.jsonl`);
}

export function persistCase(evaluatedCase: EvaluatedCase): void {
  const dir = getResultsDir();
  if (!dir) return;
  const persisted = buildTestCaseEntry(evaluatedCase, workerCaseCount++);
  fs.appendFileSync(workerFilePath(dir), JSON.stringify(persisted) + "\n");
}

export function readPersistedCases(dir: string): PersistedCase[] {
  if (!fs.existsSync(dir)) return [];
  const cases: PersistedCase[] = [];
  for (const file of fs.readdirSync(dir)) {
    if (!file.endsWith(".jsonl")) continue;
    const content = fs.readFileSync(path.join(dir, file), "utf-8");
    for (const line of content.split("\n")) {
      if (line.trim()) cases.push(JSON.parse(line) as PersistedCase);
    }
  }
  return cases;
}

function persistedToTestResult(
  { conversational, entry, metricsData }: PersistedCase,
  index: number,
): TestResult {
  return {
    name: (entry.name as string) ?? `test_case_${index}`,
    success: entry.success as boolean,
    metricsData,
    conversational,
    index,
    input: entry.input as string | undefined,
    actualOutput: entry.actualOutput as string | undefined,
    expectedOutput: entry.expectedOutput as string | undefined,
    context: entry.context as string[] | undefined,
    retrievalContext: entry.retrievalContext as string[] | undefined,
    turns: entry.turns as TestResult["turns"],
  };
}

export interface WrapUpOptions {
  runDuration?: number;
  official?: boolean;
  identifier?: string;
  printResults?: boolean;
  display?: TestRunDisplay;
}

export async function wrapUpTestRun(
  dir: string,
  options: WrapUpOptions = {},
): Promise<{ link: string | null; testRunId: string | null }> {
  const cases = readPersistedCases(dir);
  if (cases.length === 0) return { link: null, testRunId: null };

  // Written by `logHyperparameters()` in whichever worker ran the test file.
  const hyperparameters = readHyperparameters(dir);

  if (options.printResults ?? true) {
    const display = options.display ?? getDisplay();
    const results = cases
      .map(persistedToTestResult)
      .filter(
        (result) =>
          display === "all" ||
          (display === "passing" ? result.success : !result.success),
      );
    // Every case still gets posted; `--display` only filters the terminal.
    printResultsTable(results);
    printHyperparametersWarning(hyperparameters);
  }

  const runDuration = options.runDuration ?? 0;
  const official = options.official ?? false;
  const run: LocalTestRun = {
    link: null,
    savedAt: new Date().toISOString(),
    runDuration,
    official,
    identifier: options.identifier,
    hyperparameters,
    ...summarizeCases(cases),
    cases,
  };
  // Saved before posting so `deepeval view` can upload it later if the post
  // fails or there is no API key yet.
  saveLatestTestRun(run);

  const posted = await postPersistedTestRun(cases, runDuration, {
    official,
    identifier: options.identifier,
    hyperparameters,
  });

  run.link = posted.link;
  run.testRunId = posted.testRunId;
  saveLatestTestRun(run);
  const exported = exportTestRunJson(run);
  if (exported) console.log(`✅ Test run saved to: ${exported}`);

  return posted;
}

export function _resetWorkerCaseCount(): void {
  workerCaseCount = 0;
}

export type { MetricData };
